import copy
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest
from ghostcaddie.video.shot_people import BodyClubDetector,native_club_detail


def body():
    return {'bbox':[70,20,130,150],'track_id':3,'keypoints':{'left_wrist':{'point':[99,85],'confidence':.9},'right_wrist':{'point':[101,85],'confidence':.9}}}


def proposal():
    return {'point':[260,250],'shaft':[[200,170],[258,248]],'body_track_id':3,
            'proposal_box':[250,242,272,260],'shaft_width_px':2.,
            'support_pixels':[[260,250]],'dark_core_area':17,'provenance':'current_pixels'}


def model():
    return SimpleNamespace(predict=mock.Mock(return_value=[SimpleNamespace(boxes=[])]))


def test_native_current_pixel_coordinates_convert_once_and_do_not_mutate_pose():
    b=body();original=copy.deepcopy(b);f=np.zeros((180,240,3),np.uint8);native=np.zeros((360,480,3),np.uint8)
    def supported(image,current,proposals,**kw):
        assert image is native
        assert current['keypoints']['left_wrist']['point']==[198,170]
        assert current['bbox']==[140,40,260,300]
        return proposal()
    with mock.patch('ghostcaddie.video.shot_people.supported_club',side_effect=supported),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',return_value=None):
        head=native_club_detail(f,b,native,model())
    assert b==original
    assert head['point']==[130,125]
    assert head['shaft']==[[100,85],[129,124]]
    assert head['proposal_box']==[125,121,136,130]
    assert head['support_pixels']==[[130,125]] and head['shaft_width_px']==1
    assert head['dark_core_area']==17 and head['core_area_unit']=='native_source_pixels_squared'


@pytest.mark.parametrize('kind',['learned','geometric','disagreement'])
def test_native_ambiguity_or_shaft_head_disagreement_cannot_publish(kind):
    f=np.zeros((180,240,3),np.uint8);native=np.zeros((360,480,3),np.uint8)
    def learned(*args,diagnostics,**kwargs):
        diagnostics['ambiguous']=kind=='learned';return proposal()
    def geometric(*args,diagnostics,**kwargs):
        diagnostics['ambiguous']=kind=='geometric';other=proposal();other['point']=[350,300];return other if kind=='disagreement' else None
    with mock.patch('ghostcaddie.video.shot_people.supported_club',side_effect=learned),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',side_effect=geometric):
        assert native_club_detail(f,body(),native,model()) is None


@pytest.mark.parametrize('shape',[(180,240),(360,500)])
def test_native_search_requires_downscaled_uniform_source(shape):
    m=model();f=np.zeros((180,240,3),np.uint8)
    assert native_club_detail(f,body(),np.zeros((*shape,3),np.uint8),m) is None
    m.predict.assert_not_called()


def test_missing_current_hands_cannot_run_native_proposal_search():
    b=body();b['keypoints']={};m=model()
    assert native_club_detail(np.zeros((180,240,3),np.uint8),b,np.zeros((360,480,3),np.uint8),m) is None
    m.predict.assert_not_called()


@pytest.mark.parametrize('ambiguous',[True,False])
def test_baseline_ambiguity_is_preserved_and_baseline_head_stays_separate(ambiguous):
    detector=BodyClubDetector.__new__(BodyClubDetector);detector.head=model()
    def learned(*args,diagnostics,**kwargs):diagnostics['ambiguous']=ambiguous;return None
    with mock.patch('ghostcaddie.video.shot_people.supported_club',side_effect=learned),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',return_value=None),mock.patch('ghostcaddie.video.shot_people.native_club_detail',return_value=proposal()) as detail:
        result=detector._club(np.zeros((180,240,3),np.uint8),body(),np.zeros((360,480,3),np.uint8))
    assert detector.baseline_head is None
    assert (result is None)==ambiguous
    assert detail.call_count==(0 if ambiguous else 1)


def test_existing_head_is_authoritative_and_does_not_run_native_search():
    detector=BodyClubDetector.__new__(BodyClubDetector);detector.head=model();old=proposal()
    with mock.patch('ghostcaddie.video.shot_people.supported_club',return_value=old),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',return_value=None),mock.patch('ghostcaddie.video.shot_people.native_club_detail') as detail:
        assert detector._club(np.zeros((180,240,3),np.uint8),body(),None) is old
    assert detector.baseline_head is old
    detail.assert_not_called()


def test_geometric_only_native_shoe_lobe_cannot_publish():
    with mock.patch('ghostcaddie.video.shot_people.supported_club',return_value=None),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',return_value=proposal()):
        assert native_club_detail(np.zeros((180,240,3),np.uint8),body(),np.zeros((360,480,3),np.uint8),model()) is None


def test_baseline_disagreement_cannot_be_rescued_by_native_search():
    detector=BodyClubDetector.__new__(BodyClubDetector);detector.head=model();other=proposal();other['point']=[350,300]
    with mock.patch('ghostcaddie.video.shot_people.supported_club',return_value=proposal()),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',return_value=other),mock.patch('ghostcaddie.video.shot_people.native_club_detail') as detail:
        assert detector._club(np.zeros((180,240,3),np.uint8),body(),np.zeros((360,480,3),np.uint8)) is None
    assert detector.baseline_head is None
    detail.assert_not_called()


def test_mixed_resolution_head_points_do_not_create_velocity():
    from ghostcaddie.video.shot_metrics import analyze_motion
    b=body();b['keypoints'].update({name:{'point':point,'confidence':.9} for name,point in [('left_shoulder',[95,40]),('right_shoulder',[105,40]),('left_hip',[95,90]),('right_hip',[105,90])]})
    rows=[]
    for n in range(3):
        c=proposal();c['point']=[130+n*5,125];c['search_resolution']='native_source_pixels' if n else 'analysis_pixels'
        rows.append({'frame':n,'scene':0,'body_scene':0,'timestamp_seconds':n*.04,'view':'ground','body':copy.deepcopy(b),'clubhead':c,'ball':None,'target':None,'previous_to_current':None})
    motion=analyze_motion(rows,source_sha256='a'*64)
    for name in ['clubhead_image','clubhead_body_relative','clubhead_torso_relative']:
        assert motion['motion'][name]['samples']==1


def test_current_round_white_ball_can_veto_head_shadow_without_ball_detection():
    import cv2
    from ghostcaddie.video.shot_people import round_bright_head_conflict
    frame=np.full((200,200,3),(60,125,70),np.uint8)
    cv2.circle(frame,(100,100),5,(245,245,245),-1)
    assert round_bright_head_conflict(frame,[100,106],300)
    assert not round_bright_head_conflict(frame,[116,100],300)


def test_long_chrome_strip_and_boundary_white_background_do_not_create_round_conflict():
    import cv2
    from ghostcaddie.video.shot_people import round_bright_head_conflict
    frame=np.full((200,200,3),(60,125,70),np.uint8)
    cv2.rectangle(frame,(85,98),(112,101),(245,245,245),-1)
    assert not round_bright_head_conflict(frame,[100,105],500)
    assert not round_bright_head_conflict(np.full_like(frame,245),[100,105],500)


def test_native_supported_semantic_head_still_withholds_current_ball_shadow():
    import cv2
    frame=np.zeros((180,240,3),np.uint8);native=np.full((360,480,3),(60,125,70),np.uint8)
    cv2.circle(native,(260,244),5,(245,245,245),-1)
    with mock.patch('ghostcaddie.video.shot_people.supported_club',return_value=proposal()),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',return_value=None):
        assert native_club_detail(frame,body(),native,model()) is None


@pytest.mark.parametrize('channel',['learned','geometric'])
def test_baseline_core_ambiguity_blocks_native_rescue(channel):
    detector=BodyClubDetector.__new__(BodyClubDetector);detector.head=model()
    def ambiguous(*args,diagnostics,**kw):diagnostics['core_ambiguous']=True;return None
    with mock.patch('ghostcaddie.video.shot_people.supported_club',side_effect=ambiguous if channel=='learned' else None,return_value=None),mock.patch('ghostcaddie.video.shot_people.shaft_search_club',side_effect=ambiguous if channel=='geometric' else None,return_value=None),mock.patch('ghostcaddie.video.shot_people.native_club_detail') as detail:
        assert detector._club(np.zeros((180,240,3),np.uint8),body(),np.zeros((360,480,3),np.uint8)) is None
    detail.assert_not_called()


def test_dark_lobe_tie_exports_ambiguity_without_changing_baseline_result():
    import cv2
    from ghostcaddie.video.shot_people import compact_head
    frame=np.full((200,200,3),140,np.uint8)
    cv2.rectangle(frame,(93,99),(96,102),(20,20,20),-1)
    cv2.rectangle(frame,(104,99),(107,102),(20,20,20),-1)
    diagnostics={}
    assert compact_head(frame,[90,95,110,108],[100,100],300,diagnostics=diagnostics) is None
    assert diagnostics['core_ambiguous']


def test_lower_value_current_ball_and_recorded_negative_evidence():
    import cv2
    from ghostcaddie.video.shot_people import round_bright_head_conflict
    frame=np.full((200,200,3),(60,125,70),np.uint8);cv2.circle(frame,(100,100),5,(175,175,175),-1);d={}
    assert round_bright_head_conflict(frame,[100,106],300,diagnostics=d)
    assert d['nearest']['peak_value']==175 and d['nearest']['role'].startswith('negative')
    assert d['nearest']['distance_over_diameter']<.75


def test_native_ball_overlap_is_more_conservative_than_baseline_without_changing_baseline():
    from ghostcaddie.video.automatic_shot import head_ball_overlap
    b={'point':[100,100],'size':[20,20]};c={'point':[112,100]}
    assert not head_ball_overlap(c,b)
    assert head_ball_overlap(dict(c,search_resolution='native_source_pixels'),b)


def test_rounding_nonuniform_source_cannot_enter_single_scale_native_export():
    m=model()
    assert native_club_detail(np.zeros((180,240,3),np.uint8),body(),np.zeros((720,961,3),np.uint8),m) is None
    m.predict.assert_not_called()


def test_mixed_resolution_distributions_are_split_instead_of_pooled():
    from ghostcaddie.video.shot_metrics import analyze_motion
    b=body();b['keypoints'].update({n:{'point':p,'confidence':.9} for n,p in [('left_shoulder',[95,40]),('right_shoulder',[105,40]),('left_hip',[95,90]),('right_hip',[105,90])]})
    rows=[]
    for n in range(4):
        c=proposal();c['point']=[130+n*5,125];c['search_resolution']='native_source_pixels' if n>=2 else 'analysis_pixels'
        rows.append({'frame':n,'scene':0,'body_scene':0,'timestamp_seconds':n*.04,'view':'ground','body':copy.deepcopy(b),'clubhead':c,'ball':None,'target':None,'previous_to_current':None})
    a=analyze_motion(rows,source_sha256='a'*64);g=a['segments'][0]
    assert a['motion']['clubhead_image']['samples']==0
    assert g['club_mixed_resolution_summary_withheld'] and g['motion']['clubhead_image']['samples']==0
    assert g['club_motion_by_search_resolution']['analysis_pixels']['clubhead_image']['samples']==1
    assert g['club_motion_by_search_resolution']['native_source_pixels']['clubhead_image']['samples']==1
