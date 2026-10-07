import copy
import pytest
from ghostcaddie.video.four_object_evaluation import evaluate_four_objects
SHA='a'*64

def fixture():
 objects={n:{'visibility':'unreviewed','label_source':'unreviewed'} for n in ['ball','clubhead','body','cup']}
 labels={'schema_version':'fairwayos-four-object-labels.v1','source_sha256':SHA,'width':100,'height':100,'frames':[{'frame':0,'timestamp_seconds':0.,'objects':objects}]}
 return labels,[{'frame':0,'timestamp_seconds':0.}]

def score(l,r):
 for row in r:
  for name in ['ball','clubhead','body','target']:
   if isinstance(row.get(name),dict):
    row[name].setdefault('state','observed');row[name].setdefault('provenance','current_pixels')
 return evaluate_four_objects(l,r,source_sha256=SHA,native_pts=[0.])

def visible(l,name):l['frames'][0]['objects'][name]={'visibility':'visible','label_source':'human_ground_truth','point':[10.,10.],'tolerance_px':3.,'confusers':[[70.,70.,90.,90.]]}

def test_unreviewed_withholds_metrics():
 l,r=fixture();x=score(l,r);assert not x['metrics_available'];assert all(v['visible_recall'] is None for v in x['objects'].values())

@pytest.mark.parametrize('name,key',[('ball','ball'),('clubhead','clubhead'),('cup','target')])
def test_current_position_and_absence(name,key):
 l,r=fixture();visible(l,name);r[0][key]={'point':[11.,11.],'provenance':'current_pixels'}
 assert score(l,r)['objects'][name]['visible_recall']==1
 l['frames'][0]['objects'][name]={'visibility':'absent','label_source':'human_confirmed'}
 assert score(l,r)['objects'][name]['false_positive_rate_on_absent']==1

@pytest.mark.parametrize('location,wrong', [([80.,80.],1),([45.,45.],0)])
def test_localization_error_is_not_automatically_wrong_lock(location,wrong):
 l,r=fixture();visible(l,'ball');r[0]['ball']={'point':location};o=score(l,r)['objects']['ball'];assert o['counts']['localization_errors']==1-wrong;assert o['counts']['wrong_locks']==wrong;assert o['visible_recall']==0;assert o['center_error_all_visible_predictions']['samples']==1

@pytest.mark.parametrize('source',['pseudo_label','model_prediction','ai_review'])
def test_ai_labels_cannot_be_independent_reference(source):
 l,r=fixture();visible(l,'ball');l['frames'][0]['objects']['ball']['label_source']=source
 with pytest.raises(ValueError):score(l,r)

def test_source_and_pts_binding():
 l,r=fixture();l['source_sha256']='b'*64
 with pytest.raises(ValueError):score(l,r)
 l['source_sha256']=SHA;l['frames'][0]['timestamp_seconds']=.03
 with pytest.raises(ValueError):score(l,r)

@pytest.mark.parametrize('pred',[{'point':[10.,10.],'state':'interpolated'},{'point':[10.,10.],'provenance':'held_previous'}])
def test_no_held_or_interpolated_credit(pred):
 l,r=fixture();visible(l,'ball');r[0]['ball']=pred;assert score(l,r)['objects']['ball']['counts']['misses']==1

def test_body_identity_and_anchor_error_separate():
 l,r=fixture();l['frames'][0]['objects']['body']={'visibility':'visible','label_source':'human_ground_truth','bbox':[1.,1.,40.,90.],'anchors':{'left_wrist':[10.,20.]},'tolerance_px':3.};r[0]['body']={'bbox':[1.,1.,40.,90.],'keypoints':{'left_wrist':{'point':[20.,20.]}}}
 o=score(l,r)['objects']['body'];assert o['visible_recall']==1;assert o['body_anchors']['left_wrist']['within_tolerance']==0;assert o['body_anchors']['left_wrist']['center_error']['median_px']==10

def test_ambiguous_not_absent():
 l,r=fixture();l['frames'][0]['objects']['cup']={'visibility':'ambiguous','label_source':'unreviewed'};r[0]['target']=[10.,10.];o=score(l,r)['objects']['cup'];assert o['scored_frames']==0;assert o['counts']['false_on_absent']==0

@pytest.mark.parametrize('mutation',[lambda l:l['frames'].append(copy.deepcopy(l['frames'][0])), lambda l:l['frames'][0]['objects'].pop('cup'),lambda l:l['frames'][0]['objects']['ball'].update(visibility='absent',point=[10.,10.],label_source='human_confirmed')])
def test_reject_invalid_labels(mutation):
 l,r=fixture();mutation(l)
 with pytest.raises(ValueError):score(l,r)


def test_missing_body_does_not_hide_visible_anchor_miss():
 l,r=fixture();l['frames'][0]['objects']['body']={'visibility':'visible','label_source':'human_ground_truth','bbox':[1.,1.,40.,90.],'anchors':{'left_wrist':[10.,20.]},'tolerance_px':3.}
 o=score(l,r)['objects']['body'];assert o['counts']['misses']==1;assert o['body_anchors']['left_wrist']['visible']==1;assert o['body_anchors']['left_wrist']['observed']==0

@pytest.mark.parametrize('timestamp',[float('nan'),float('inf'),True])
def test_nonfinite_observation_pts_rejected(timestamp):
 l,r=fixture();r[0]['timestamp_seconds']=timestamp
 with pytest.raises(ValueError):score(l,r)

from ghostcaddie.video.four_object_evaluation import validate_source_groups

@pytest.mark.parametrize('duplicate_bytes',[False,True])
def test_reject_session_or_same_bytes_leak(duplicate_bytes):
 rows=[{'id':'a','source_group':'shoot-a','source_sha256':SHA,'split':'train'}, {'id':'b','source_group':'shoot-b' if duplicate_bytes else 'shoot-a','source_sha256':SHA if duplicate_bytes else 'b'*64,'split':'test'}]
 with pytest.raises(ValueError):validate_source_groups(rows)

def test_consumed_group_cannot_be_independent_test():
 rows=[{'id':'a','source_group':'masters2019','source_sha256':SHA,'split':'holdout'}]
 with pytest.raises(ValueError):validate_source_groups(rows,consumed_groups=['masters2019'])

def test_siblings_together_are_valid():
 rows=[{'id':'a','source_group':'new-shoot','source_sha256':SHA,'split':'development'}, {'id':'b','source_group':'new-shoot','source_sha256':'b'*64,'split':'development'}]
 assert validate_source_groups(rows)['group_isolation_verified']


@pytest.mark.parametrize('prediction',[{'point':[10.,10.]}, {'point':[10.,10.],'state':'observed'}, [10.,10.]])
def test_no_unprovenanced_credit(prediction):
 l,r=fixture();visible(l,'ball');r[0]['ball']=prediction
 o=evaluate_four_objects(l,r,source_sha256=SHA,native_pts=[0.])['objects']['ball'];assert o['counts']['misses']==1

def test_anchor_distance_and_target_person_correctness_are_separate():
 l,r=fixture();l['frames'][0]['objects']['body']={'visibility':'visible','label_source':'human_ground_truth','bbox':[1.,1.,40.,90.],'anchors':{'left_wrist':[10.,20.]},'tolerance_px':3.};r[0]['body']={'bbox':[50.,1.,90.,90.],'keypoints':{'left_wrist':{'point':[10.,20.]}}}
 o=score(l,r)['objects']['body'];a=o['body_anchors']['left_wrist'];assert o['visible_recall']==0;assert a['within_tolerance']==1;assert a['target_person_within_tolerance']==0
