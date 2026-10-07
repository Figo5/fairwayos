import unittest
from unittest import mock
from pathlib import Path
import tempfile
import cv2
import numpy as np
from ghostcaddie.video.temporal_objects import sky_candidates,cup_candidates,unique_chains,cup_publication_runs
from ghostcaddie.video.automatic_shot import observe_video

class TemporalObjectTests(unittest.TestCase):
    def rows(self,two=False,cut=False,missing=False):
        rows=[]
        for n in range(30):
            proposals=[{'point':[100.+n,100.],'size':[4,4]}]
            if two:proposals.append({'point':[230.+n,100.],'size':[4,4]})
            if missing and n==15:proposals=[]
            rows.append({'frame':n,'scene':n//10 if cut else 0,'p':proposals})
        return rows
    def test_competing_sustained_sky_objects_are_withheld(self):
        c=unique_chains(self.rows(two=True),'p',min_support=12,radius=15,moving=True)
        self.assertTrue(all(not chain for chain in c))
    def test_a_cut_never_connects_subthreshold_tracks(self):
        self.assertEqual(unique_chains(self.rows(cut=True),'p',min_support=12,radius=15,moving=True),[])
    def test_missing_components_are_never_filled(self):
        c=unique_chains(self.rows(missing=True),'p',min_support=12,radius=15,moving=True)
        self.assertNotIn(15,[r['frame'] for chain in c for r,p in chain])
    def test_sky_requires_contrast_and_a_sky_surround(self):
        im=np.full((240,320,3),220,np.uint8);cv2.circle(im,(140,100),2,(170,170,170),-1)
        self.assertEqual(len(sky_candidates(im)),1)
        self.assertEqual(sky_candidates(np.full_like(im,(60,125,70))),[])
        self.assertEqual(sky_candidates(np.full_like(im,220)),[])
    def test_flag_free_rim_excludes_round_ball_and_person(self):
        im=np.full((240,320,3),(60,125,70),np.uint8)
        cv2.ellipse(im,(100,160),(15,4),0,0,360,(235,235,235),-1)
        cv2.circle(im,(220,170),8,(240,240,240),-1)
        self.assertEqual(len(cup_candidates(im)),1)
        self.assertEqual(cup_candidates(im,[(75,120,125,190)]),[])
    def test_broad_white_sand_shape_is_not_a_flag_free_cup(self):
        im=np.full((720,1280,3),(60,125,70),np.uint8)
        cv2.ellipse(im,(640,450),(65,7),0,0,360,(235,235,235),-1)
        proposals=cup_candidates(im)
        self.assertEqual(len(proposals),1)
        self.assertFalse(proposals[0]['publishable_cup'])
        frames=[(n,im.copy()) for n in range(25)]
        with tempfile.TemporaryDirectory() as d,mock.patch('ghostcaddie.video.automatic_shot.decoded_frames',return_value=iter(frames)):
            obs=observe_video('unlabelled.mp4',{'width':1280,'height':720,'frames':25},Path(d),lambda frame:[])
        self.assertTrue(all(r['target'] is None for r in obs['frames']))
    def test_broad_rival_cannot_promote_a_narrow_marker(self):
        im=np.full((240,320,3),(60,125,70),np.uint8)
        cv2.ellipse(im,(100,160),(15,4),0,0,360,(235,235,235),-1)
        cv2.ellipse(im,(220,160),(25,4),0,0,360,(235,235,235),-1)
        proposals=cup_candidates(im)
        self.assertEqual(sorted(p['publishable_cup'] for p in proposals),[False,True])
        frames=[(n,im.copy()) for n in range(25)]
        with tempfile.TemporaryDirectory() as d,mock.patch('ghostcaddie.video.automatic_shot.decoded_frames',return_value=iter(frames)):
            obs=observe_video('unlabelled.mp4',{'width':320,'height':240,'frames':25},Path(d),lambda frame:[])
        self.assertTrue(all(r['target'] is None for r in obs['frames']))
    def test_dark_sky_object_uses_bright_surround_not_bright_center(self):
        im=np.full((240,320,3),220,np.uint8)
        cv2.circle(im,(140,100),2,(70,70,70),-1)
        self.assertEqual(len(sky_candidates(im)),1)
    def test_nonconsecutive_frames_do_not_form_one_track(self):
        rows=self.rows();rows=[r for r in rows if r['frame']%2==0]
        self.assertEqual(unique_chains(rows,'p',min_support=12,radius=15,moving=True),[])
    def test_moving_flat_object_is_not_a_stationary_cup(self):
        frames=[]
        for n in range(25):
            im=np.full((240,320,3),(60,125,70),np.uint8)
            cv2.ellipse(im,(70+n*5,160),(15,4),0,0,360,(235,235,235),-1)
            frames.append((n,im))
        with tempfile.TemporaryDirectory() as d,mock.patch('ghostcaddie.video.automatic_shot.decoded_frames',return_value=iter(frames)):
            obs=observe_video('unlabelled.mp4',{'width':320,'height':240,'frames':25},Path(d),lambda frame:[])
        self.assertTrue(all(r['target'] is None for r in obs['frames']))
    def test_lost_ball_requires_new_neural_evidence_to_reacquire(self):
        frames=[]
        for n in range(36):
            im=np.full((240,320,3),(60,125,70),np.uint8)
            if n<12 or n>16:cv2.circle(im,(210 if n<12 else 240,165),9,(245,245,245),-1)
            frames.append((n,im))
        calls=[0]
        def detector(frame):
            n=calls[0]*6;calls[0]+=1
            if 12<=n<=16:return []
            x=210 if n<12 else 240
            return [{'class_id':32,'score':.95,'box':[x-9,156,x+9,174]}]
        with tempfile.TemporaryDirectory() as d,mock.patch('ghostcaddie.video.automatic_shot.decoded_frames',return_value=iter(frames)):
            obs=observe_video('unlabelled.mp4',{'width':320,'height':240,'frames':36},Path(d),detector)
        self.assertTrue(all(obs['frames'][n]['ball'] is None for n in range(12,17)))
        self.assertIsNotNone(obs['frames'][18]['ball'])
        self.assertTrue(any(i['method']=='automatic_neural_reacquisition' for i in obs['initializations']))
    def publication_fixture(self, points=None, camera=True):
        rows=[]
        for n in range(30):
            p={'point':[100. if points is None else points[n],160.],'size':[31,9],'publishable_cup':True}
            rows.append({'frame':n,'scene':0,'rim_proposals':[p],'target':None,'ball':None,
                'previous_to_current':[[1,0,0],[0,1,0]] if camera and n else None})
        return rows,[(row,row['rim_proposals'][0]) for row in rows]
    def test_fragmented_broad_rival_blocks_a_narrow_marker(self):
        rows,chain=self.publication_fixture()
        for row in rows:
            if row['frame']!=16:row['rim_proposals'].append({'point':[220.,160.],'size':[80,9],'publishable_cup':False})
        self.assertEqual(cup_publication_runs(rows,[chain]),[])
    def test_scattered_stationary_pairs_are_not_a_supported_cup(self):
        rows,chain=self.publication_fixture([100.+6*((n//2)%2) for n in range(30)])
        self.assertEqual(cup_publication_runs(rows,[chain]),[])
    def test_camera_followed_flat_object_is_not_a_registered_cup(self):
        rows,chain=self.publication_fixture()
        for row in rows[1:]:row['previous_to_current']=[[1,0,8],[0,1,0]]
        self.assertEqual(cup_publication_runs(rows,[chain]),[])
    def test_failed_registration_withholds_appearance_only_cup(self):
        rows,chain=self.publication_fixture(camera=False)
        self.assertEqual(cup_publication_runs(rows,[chain]),[])
    def test_existing_flag_target_is_preserved(self):
        rows,chain=self.publication_fixture()
        for row in rows:row['target']=[220.,200.]
        self.assertEqual(cup_publication_runs(rows,[chain]),[])
        self.assertTrue(all(row['target']==[220.,200.] for row in rows))
    def test_ball_overlap_withholds_flat_rim(self):
        rows,chain=self.publication_fixture()
        for row in rows:row['ball']={'point':[100.,160.],'size':[10,10]}
        self.assertEqual(cup_publication_runs(rows,[chain]),[])
    def test_publication_starts_at_first_contiguous_supported_frame(self):
        rows,chain=self.publication_fixture()
        for row in rows[:5]:row['previous_to_current']=None
        runs=cup_publication_runs(rows,[chain])
        self.assertEqual([row['frame'] for row,p in runs[0]],list(range(4,30)))
    def test_a_cup_can_start_without_any_ball_detection(self):
        frames=[]
        for n in range(25):
            im=np.full((240,320,3),(60,125,70),np.uint8)
            cv2.ellipse(im,(100,160),(15,4),0,0,360,(235,235,235),-1)
            frames.append((n,im))
        with tempfile.TemporaryDirectory() as d,mock.patch('ghostcaddie.video.automatic_shot.decoded_frames',return_value=iter(frames)),mock.patch('ghostcaddie.video.automatic_shot.affine',return_value=(np.array([[1.,0.,0.],[0.,1.,0.]]),{'fixture':'known_static_camera'})):
            obs=observe_video('unlabelled.mp4',{'width':320,'height':240,'frames':25},Path(d),lambda frame:[])
        self.assertTrue(all(r['ball'] is None for r in obs['frames']))
        self.assertTrue(all(r['target'] is not None for r in obs['frames']))
        self.assertEqual(obs['operator_seed_count'],0)

if __name__=='__main__':unittest.main()
