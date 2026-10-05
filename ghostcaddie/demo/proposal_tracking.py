"""Causal confirmation of measured proposal cores; no synthesized coordinates."""
import copy
import math

from .shaft_head import ShaftHeadPolicy, CONFIRM_FRAMES, MAX_DELTA_720
from .tracking import STATUS_OBSERVED

METHOD = 'learned_shaft_core'
METHOD_REACQUIRED = METHOD + '_reacquired'


class ProposalHeadPolicy(ShaftHeadPolicy):
    """Original-seed replay using proposal-restricted current pixel measurements.

    Inherits the existing body validation, two-current-frame confirmation and
    terminal cut rules. The original seed remains an operator-only observation.
    """
    def step(self, frame, image, body_frame=None, *, detections):
        from .proposal_head import measure_proposals, PROFILE, _validate_detections
        if type(frame) is not int or frame < 0 or frame != self.last_frame + 1:
            raise ValueError('Frames must arrive once in contiguous forward order')
        self._validate_image(image)
        _validate_detections(detections,image.shape[1],image.shape[0])
        if self.terminal:
            self.last_frame = frame
            return self._derived(frame, 'after_structural_cut')
        if frame == self.seed['frame']:
            self.last_frame = frame
            return self._seed_step(frame)
        if frame in self.cuts:
            self.last_frame = frame
            self.terminal = True
            self.pending = None
            self.gapped = True
            return self._derived(frame, 'structural_cut_is_terminal')
        body, reason = self._read_body(frame, body_frame)
        evidence = (measure_proposals(image, [int(v) for v in self.origin],
                                      [int(v) for v in self.source_size], body, detections)
                    if body is not None else {'selected': None, 'body_fail_closed': reason})
        self.last_frame = frame
        point, raw = self._track_step(frame, body, reason, evidence)
        if point['origin'] == 'automatic':
            point['method'] = (METHOD_REACQUIRED if point['method'].endswith('_reacquired') else METHOD)
            point['x'], point['y'] = evidence['selected']['centroid_native']
            # _track_step's decision embeds the point's old method; keep it truthful.
            raw['decision'].update({k: point[k] for k in ('x', 'y', 'method')})
        raw['proposal_profile'] = copy.deepcopy(PROFILE)
        return point, raw


class UnseededConfirmation:
    """Confirm unseeded diagnostic cores without inventing an operator placement."""
    def __init__(self, source_height, start_frame=0, cut_frames=()):
        if CONFIRM_FRAMES != 2:
            raise ValueError('This confirmation policy requires the two-frame shaft profile')
        if type(source_height) is not int or source_height <= 0 or type(start_frame) is not int or start_frame < 0:
            raise ValueError('Expected positive native height and nonnegative start frame')
        if any(type(c) is not int or c < 0 for c in cut_frames):
            raise ValueError('Cuts must be nonnegative source frames')
        self.last_frame = start_frame - 1
        self.cuts = set(cut_frames)
        self.terminal = False
        self.pending = None
        self.confirmed = False
        self.last_point = None
        self.gapped = True
        self.max_delta = MAX_DELTA_720 * source_height / 720

    def step(self, frame, selected):
        if type(frame) is not int or frame != self.last_frame + 1:
            raise ValueError('Unseeded frames must be contiguous and forward')
        self.last_frame = frame
        self.terminal = self.terminal or frame in self.cuts
        xy = selected.get('centroid_native') if selected is not None else None
        if xy is not None and (len(xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)):
            raise ValueError('Selected core must contain a finite measured centroid')
        missing = {'source_frame': frame, 'x': None, 'y': None, 'score': None,
                   'status': 'missing', 'origin': 'derived', 'method': ''}
        if self.terminal or xy is None:
            self.pending = None
            self.gapped = True
            return missing
        continuing = self.last_point is not None and not self.gapped and math.dist(xy, self.last_point) <= self.max_delta
        confirming = self.pending is not None and self.pending[0] == frame - 1 and math.dist(xy, self.pending[1]) <= self.max_delta
        if not (continuing or confirming):
            self.pending = (frame, list(xy))
            self.gapped = True
            return missing
        method = METHOD_REACQUIRED if self.confirmed and not continuing else METHOD
        self.pending = None
        self.last_point = list(xy)
        self.gapped = False
        self.confirmed = True
        return {'source_frame': frame, 'x': xy[0], 'y': xy[1], 'score': None,
                'status': STATUS_OBSERVED, 'origin': 'automatic', 'method': method}
