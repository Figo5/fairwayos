import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
import urllib.error
from unittest import mock
from types import SimpleNamespace

from ghostcaddie.upload.server import build_server, _run_automatic_shot_job
from ghostcaddie.upload.jobs import JobStore
from ghostcaddie.upload.automatic_ui import PAGE


class UploadOnlyTests(unittest.TestCase):
    def test_cancelled_job_never_launches_or_publishes_a_worker(self):
        with tempfile.TemporaryDirectory() as d:
            store=JobStore(d);job=store.create('clip.mp4');store.cancel(job.id)
            with mock.patch('ghostcaddie.upload.server.subprocess.Popen') as launch:
                _run_automatic_shot_job(store,job.id,SimpleNamespace(path='clip.mp4',sha256='abc'))
            launch.assert_not_called()
            store.start(job.id);store.finish(job.id,{});store.fail(job.id,'late failure')
            self.assertEqual(store.get(job.id).state,'cancelled')
            self.assertIsNone(store.get(job.id).result)

    def test_cancellation_during_launch_stops_the_new_worker(self):
        with tempfile.TemporaryDirectory() as d:
            store=JobStore(d);job=store.create('clip.mp4');process=mock.Mock()
            def launch(*args,**kwargs):
                store.cancel(job.id)
                return process
            with mock.patch('ghostcaddie.upload.server.subprocess.Popen',side_effect=launch), mock.patch.object(store,'kill_proc') as kill:
                _run_automatic_shot_job(store,job.id,SimpleNamespace(path='clip.mp4',sha256='abc'))
            kill.assert_called_once_with(job.id)
            process.communicate.assert_not_called()
            self.assertEqual(store.get(job.id).state,'cancelled')
            self.assertIsNone(store.get(job.id).result)
            self.assertNotIn(job.id,store._procs)

    def test_result_for_another_source_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as d:
            store=JobStore(d);job=store.create('clip.mp4');process=mock.Mock(returncode=0)
            process.communicate.return_value=(json.dumps({'source':{'sha256':'wrong'}}),'')
            with mock.patch('ghostcaddie.upload.server.subprocess.Popen',return_value=process):
                _run_automatic_shot_job(store,job.id,SimpleNamespace(path='clip.mp4',sha256='abc'))
            self.assertEqual(store.get(job.id).state,'failed')
            self.assertIn('another source',store.get(job.id).error)
            self.assertIsNone(store.get(job.id).result)

    def test_product_page_starts_analysis_from_the_file_input_without_seed_controls(self):
        self.assertIn("$('clip').onchange=async",PAGE)
        self.assertIn("form.append('analysis','shot_intelligence')",PAGE)
        for manual in ['pickBall','sendSeed','freshStart','frame_start','clubhead box']:
            self.assertNotIn(manual,PAGE)
        self.assertIn('Download clip',PAGE)
        self.assertIn('Original footage',PAGE)

    def test_service_accepts_video_only_job_and_attaches_verified_result(self):
        with tempfile.TemporaryDirectory() as d:
            server=build_server(d,port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_address[1]}'
            imports=server.RequestHandlerClass.store.import_dir
            source=os.path.join(imports,'clip.mp4')
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=green:size=320x240:rate=30:duration=0.4',source],check=True)
            video=os.path.join(d,'result.mp4');open(video,'wb').write(b'local-render')
            from ghostcaddie.upload.validation import validate_upload
            validated=validate_upload(source)
            analytics={'source_sha256':validated.sha256,'physical':{'ball_mph':None,'clubhead_mph':None}}
            result={'source':{'sha256':validated.sha256},'automatic_no_seed':True,'operator_seed_count':0,'verification':{'ok':True},'outputs':{'video':video},'analytics':analytics,'layers':{'ball':{'state':'partial','frames_with_observation':4},'cup':{'state':'partial','frames_with_observation':4}}}
            process=mock.Mock();process.communicate.return_value=(json.dumps(result),'');process.returncode=0
            try:
                with mock.patch('ghostcaddie.upload.server.subprocess.Popen',return_value=process) as launch:
                    # Validate the file before mocking the worker process, since
                    # validation also deliberately runs an external decoder.
                    from ghostcaddie.upload.validation import validate_upload
                    with mock.patch('ghostcaddie.upload.server.validate_upload') as validate:
                        validate.return_value=validated
                        headers={'X-FairwayOS-Token':server.RequestHandlerClass.csrf_token,'Content-Type':'application/x-www-form-urlencoded'}
                        request=urllib.request.Request(base+'/jobs',data=urllib.parse.urlencode({'name':'clip.mp4','analysis':'shot_intelligence'}).encode(),headers=headers)
                        with urllib.request.urlopen(request) as response:
                            self.assertEqual(response.status,202);job=json.load(response)
                        for _ in range(100):
                            with urllib.request.urlopen(base+'/jobs/'+job['id']) as response:state=json.load(response)
                            if state['state'] in ('done','failed'):break
                            time.sleep(.02)
                self.assertEqual(state['state'],'done')
                self.assertEqual(state['awaiting'],[])
                self.assertEqual(state['result']['operator_seed_count'],0)
                args=launch.call_args.args[0]
                self.assertIn('ghostcaddie.video.automatic_shot',args)
                self.assertNotIn('--seed',args)
                with urllib.request.urlopen(base+'/') as response:page=response.read().decode()
                self.assertIn('Automatic shot analysis',page)
                self.assertNotIn('__FAIRWAYOS_TOKEN__',page)
                with urllib.request.urlopen(base+'/result-video?job='+job['id']) as response:self.assertEqual(response.read(),b'local-render')
                with urllib.request.urlopen(base+'/result-analytics?job='+job['id']) as response:self.assertEqual(json.load(response),analytics)
                server.RequestHandlerClass.store.revoke_media(job['id'],reason='test')
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(base+'/result-analytics?job='+job['id'])
                self.assertEqual(error.exception.code,410)
            finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
