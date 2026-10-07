import copy,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
import cv2,numpy as np
from ghostcaddie.video.shot_semantics import (MODEL,SCHEMA,PROMPT,SemanticBallClient,component_images,parse_response,semantic_chains)
from ghostcaddie.video.shot_metrics import analyze_motion
from tests.test_shot_ball_detail import chain
from tests.test_shot_people_metrics import body,row


def decision(ident='c0',kind='ball'):
    return {'id':ident,'classification':kind,'visually_separable':True,'separate_from_shoe_or_head':True,'reason':'Round sphere visibly detached from the shoe.'}

def response(nonce='bound',records=None):
    return json.dumps({'schema_version':SCHEMA,'nonce':nonce,'decisions':records or [decision()]})

def rows():
    rs=chain()
    for r in rs:
        r['view']='ground';r['detail_ball_candidates'][0]['neural_checked']=r['frame']%6==0
        r['detail_ball_candidates'][0]['neural_score']=None
    return rs


class SemanticSchemaTests(unittest.TestCase):
    def test_exact_envelope_ids_types_and_no_coordinates(self):
        self.assertEqual(parse_response(response(),'bound',['c0'])['c0']['classification'],'ball')
        obj=json.loads(response())
        bad=[]
        for mutation in ('nonce','schema','coords','id','duplicate','boolean','class','extra','missing'):
            d=copy.deepcopy(obj)
            if mutation=='nonce':d['nonce']='foreign'
            elif mutation=='schema':d['schema_version']='other'
            elif mutation=='coords':d['decisions'][0]['point']=[1,2]
            elif mutation=='id':d['decisions'][0]['id']='title-18'
            elif mutation=='duplicate':d['decisions']*=2
            elif mutation=='boolean':d['decisions'][0]['visually_separable']='true'
            elif mutation=='class':d['decisions'][0]['classification']='probably_ball'
            elif mutation=='extra':d['coordinates']=[]
            else:d['decisions']=[]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):parse_response(json.dumps(d),'bound',['c0'])
        with self.assertRaises(ValueError):parse_response(response().replace('"nonce": "bound"','"nonce": "foreign", "nonce": "bound"'),'bound',['c0'])
        with self.assertRaises(ValueError):parse_response('```json\n'+response()+'\n```','bound',['c0'])


class SemanticWindowTests(unittest.TestCase):
    def test_independent_channel_keeps_neural_misses_and_current_coordinates(self):
        rs=rows();before=copy.deepcopy(rs);ds={n:decision() for n in (0,6,12)}
        found=semantic_chains(rs,decisions=ds)
        self.assertEqual([r['frame'] for r,c in found],list(range(13)))
        self.assertEqual(rs,before)
        self.assertTrue(all(c['point']==[190.,211.] for r,c in found))
        self.assertEqual(semantic_chains(rs,decisions={0:decision(),6:decision()}),[])

    def test_negative_uncertain_missing_provider_and_unseparable_reset(self):
        for variant in ('non_ball','uncertain','missing','failure','attached'):
            rs=rows();ds={n:decision() for n in (0,6,12,18)}
            if variant=='missing':del ds[6]
            elif variant=='failure':ds[6]={'status':'provider_failure'}
            elif variant=='attached':ds[6]['separate_from_shoe_or_head']=False
            else:ds[6]['classification']=variant
            with self.subTest(variant=variant):self.assertEqual(semantic_chains(rs,decisions=ds),[])

    def test_cut_view_identity_rival_missing_and_scale_are_not_bridged(self):
        for variant in ('cut','view','identity','rival','missing','scale','drift'):
            rs=rows();ds={n:decision() for n in (0,6,12,18)}
            if variant=='rival':rs[8]['detail_ball_candidates']*=2
            elif variant=='missing':rs[8]['detail_ball_candidates']=[]
            elif variant=='scale':rs[8]['detail_ball_candidates'][0]['body_height']*=1.2
            elif variant=='drift':
                for r in rs:r['detail_ball_candidates'][0]['point'][0]+=r['frame']
            else:
                for r in rs[8:]:
                    if variant=='cut':r['body_scene']=1
                    elif variant=='view':r['view']='unresolved'
                    else:r['body']['track_id']=2
            with self.subTest(variant=variant):self.assertEqual(semantic_chains(rs,decisions=ds),[])

    def test_native_camera_translation_converts_once_and_does_not_generate_points(self):
        rs=rows()
        for r in rs:
            r['detail_ball_candidates'][0]['point'][0]+=r['frame']*5
            r['previous_to_current']=[[1,0,2.5],[0,1,0]] if r['frame'] else None
        out=semantic_chains(rs,decisions={n:decision() for n in (0,6,12)},analysis_scale=.5)
        self.assertEqual(out[-1][1]['point'],[250.,211.])
        self.assertEqual(semantic_chains(rs,decisions={n:decision() for n in (0,6,12)}),[])


class SemanticProviderTests(unittest.TestCase):
    def make_item(self,p,n=0):
        im=np.full((260,300,3),(60,125,70),np.uint8);cv2.circle(im,(190,211),4,(245,245,245),-1)
        return component_images(im,{'point':[190.,211.],'size':[9,9]},p/'images',source_sha256='a'*64,frame_number=n,body_height=200)

    def fake_process(self,command,**kwargs):
        self.assertNotIn('start_new_session',kwargs);self.assertNotIn('preexec_fn',kwargs)
        self.assertEqual(command[command.index('--tools')+1],'')
        self.assertIn('--safe-mode',command)
        proc=mock.Mock(returncode=0)
        def communicate(raw,**kwargs):
            request=json.loads(raw);blocks=request['message']['content'];text=blocks[-1]['text']
            self.assertEqual(sum(b['type']=='image' for b in blocks),2)
            self.assertNotIn('a'*64,text);self.assertNotIn('point',text.split('\nnonce=')[1])
            nonce=text.split('\nnonce=')[1].split('\n')[0]
            final={'type':'result','is_error':False,'modelUsage':{MODEL:{}},'structured_output':json.loads(response(nonce))}
            return json.dumps(final)+'\n',''
        proc.communicate.side_effect=communicate
        return proc

    def test_bound_cache_reuses_only_identical_images_source_frame_prompt_model(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);item=self.make_item(p);client=SemanticBallClient(p/'client',executable='/fake/claude')
            with mock.patch('ghostcaddie.video.shot_semantics.subprocess.Popen',side_effect=self.fake_process) as proc:
                self.assertEqual(client.classify([item])[0]['status'],'completed')
                self.assertEqual(client.classify([item])[0]['status'],'cached');self.assertEqual(proc.call_count,1)
                for key,value in [('source_sha256','b'*64),('frame',1),('point',[191.,211.])]:
                    changed=copy.deepcopy(item);changed[key]=value;client.classify([changed])
                client.config['prompt_sha256']='different';client.classify([item])
                self.assertEqual(proc.call_count,5)
            cached=list((p/'client/cache').glob('*.json'))[0];cached.write_text('{broken')
            with mock.patch('ghostcaddie.video.shot_semantics.subprocess.Popen',side_effect=self.fake_process):
                # A damaged cache never grants evidence.
                client.config['prompt_sha256']=__import__('hashlib').sha256(PROMPT.encode()).hexdigest()
                self.assertIn(client.classify([item])[0]['status'],('completed','cached'))

    def test_changed_image_abstains_and_candidate_cap_is_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);item=self.make_item(p);Path(item['images'][0]['path']).write_bytes(b'changed')
            client=SemanticBallClient(p/'client',executable='/fake',max_candidates=1)
            other=copy.deepcopy(item);other['frame']=1
            with mock.patch('ghostcaddie.video.shot_semantics.subprocess.Popen') as proc:
                ds=client.classify([item,other]);self.assertEqual(ds[0]['status'],'provider_failure')
                self.assertEqual(ds[1]['status'],'budget_cap');proc.assert_not_called()
            with self.assertRaises(ValueError):client.classify([item,item])

    def test_timeout_and_malformed_result_never_publish(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);item=self.make_item(p);client=SemanticBallClient(p/'client',executable='/fake')
            proc=mock.Mock(returncode=0);proc.communicate.side_effect=[subprocess.TimeoutExpired('fake',.01),('','')]
            with mock.patch('ghostcaddie.video.shot_semantics.subprocess.Popen',return_value=proc):
                self.assertEqual(client.classify([item])[0]['status'],'provider_failure');proc.terminate.assert_called_once()
            proc.communicate.side_effect=None;proc.communicate.return_value=(json.dumps({'type':'result','is_error':False,'modelUsage':{MODEL:{}},'result':'{}'}),'')
            with mock.patch('ghostcaddie.video.shot_semantics.subprocess.Popen',return_value=proc):
                self.assertEqual(client.classify([item])[0]['status'],'provider_failure')

    def test_real_child_inherits_owned_worker_group(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);item=self.make_item(p);exe=p/'provider';pg=p/'pgid'
            exe.write_text('#!'+sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\nPath('+repr(str(pg))+').write_text(str(os.getpgrp()))\nsys.stdin.read()\nprint(json.dumps({"type":"result","is_error":True}))\n');exe.chmod(0o755)
            client=SemanticBallClient(p/'client',executable=str(exe));ds=client.classify([item])
            self.assertEqual(int(pg.read_text()),os.getpgrp());self.assertEqual(ds[0]['status'],'provider_failure')


class SemanticMotionTests(unittest.TestCase):
    def test_channels_never_pair_and_mixed_summaries_are_withheld(self):
        rs=[row(n,b=body()) for n in range(4)]
        for n,r in enumerate(rs):
            r['view']='ground';r['ball_space']='image'
            r['ball']={'point':[200+n*4,239],'body_track_id':1,'context':'near_player_native_detail' if n<2 else 'near_player_ai_semantics'}
        m=analyze_motion(rs,source_sha256='a'*64)
        self.assertEqual(m['motion']['ball_image']['samples'],0)
        self.assertTrue(m['ball_mixed_context_summary_withheld'])
        for context in ('near_player_native_detail','near_player_ai_semantics'):
            self.assertEqual(m['ball_motion_by_context'][context]['ball_image']['samples'],1)
            self.assertAlmostEqual(m['ball_motion_by_context'][context]['ball_image']['median'],100)
        self.assertTrue(all(v is None for v in m['physical'].values()))

class ActualCancellationTests(unittest.TestCase):
    def test_cancelling_owned_worker_stops_live_semantic_child(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);item=SemanticProviderTests().make_item(p);(p/'item.json').write_text(json.dumps(item));child=p/'child-pid';exe=p/'provider'
            exe.write_text('#!'+sys.executable+'\nimport os,sys,time\nfrom pathlib import Path\nPath('+repr(str(child))+').write_text(str(os.getpid()))\nsys.stdin.read()\ntime.sleep(60)\n');exe.chmod(0o755)
            code='import json;from pathlib import Path;from ghostcaddie.video.shot_semantics import SemanticBallClient;p=Path('+repr(str(p))+');SemanticBallClient(p/"client",executable=str(p/"provider")).classify([json.loads((p/"item.json").read_text())])'
            proc=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                end=time.monotonic()+10
                while not child.exists() and time.monotonic()<end:time.sleep(.03)
                self.assertTrue(child.exists());pid=int(child.read_text());self.assertEqual(os.getpgid(pid),proc.pid)
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
                end=time.monotonic()+3;alive=True
                while time.monotonic()<end:
                    state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
                    alive=bool(state) and not state.startswith('Z')
                    if not alive:break
                    time.sleep(.03)
                self.assertFalse(alive,'semantic provider survived its owning worker cancellation')
            finally:
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
