import json
import subprocess
import unittest
from unittest import TestCase

from lapps.discriminators import Uri
from mmif import AnnotationTypes, DocumentTypes, Mmif, View

import app
from app import AAPB_PUA_Kaldi


class TestAAPB_PUA_Kaldi(TestCase):

    def setUp(self) -> None:
        self.app = AAPB_PUA_Kaldi()
        self.pua_testcase = self.generate_pua_testcase()
    
    @staticmethod
    def generate_pua_testcase():
        pua_testcase = {"words": []}
        for word in zip(range(0, 20, 2), '1234567890'):
            pua_testcase['words'].append({'time': word[0], 'duration': '1.0', 'word': word[1]})
        return pua_testcase
    
    @staticmethod
    def generate_audiodocument_view(mmif:Mmif):
        ad_v = mmif.new_view()
        d = ad_v.new_annotation(DocumentTypes.AudioDocument)
        return d

    def test__align_segmentations_to_patchwork(self):
        
        v = View()
        v.id = 'tf_view'
        v.new_contain(AnnotationTypes.TimeFrame, **{'timeUnit': self.app.timeunit})
        speech = True
        last_seg_leng = 1200
        # segment_boundaries = [0.0, 10.5, 21.2, 33.2, 45.0, 58.25, 71.010]
        segment_boundaries = [0, 100, 200, 300, 400, 500]
        for i, segment_start in enumerate(segment_boundaries):
            tf = v.new_annotation(AnnotationTypes.TimeFrame)
            tf.add_property('frameType', 'speech' if speech else 'non-speech')
            tf.add_property('start', segment_start)
            tf.add_property('end', segment_boundaries[i + 1] if i < len(segment_boundaries) - 1 else segment_boundaries[
                                                                                                         -1] + last_seg_leng)
            speech = not speech
        realigned = list(zip(*self.app._align_segmentations_to_patchwork(
            v.get_annotations(AnnotationTypes.TimeFrame, frameType='speech'))))
        self.assertEqual(len(realigned), len(segment_boundaries) // 2 + len(segment_boundaries) % 2)
        gap = self.app.silence_gap * self.app.timeunit_conv[self.app.timeunit]
        prev_nend = -gap
        for i, (segid, ostart, oend, nstart, nend) in enumerate(realigned):
            self.assertEqual(ostart, segment_boundaries[i * 2])
            self.assertEqual(oend - ostart, nend - nstart)
            self.assertEqual(nstart, prev_nend + gap)
            prev_nend = nend

    def test__kaldi_to_single_textdocument(self):
        pua_testcase = self.generate_pua_testcase()
        m = Mmif(validate=False)
        d = self.generate_audiodocument_view(m)
        kaldi_view = m.new_view()
        kaldi_view.new_contain(AnnotationTypes.TimeFrame, **{'timeUnit': self.app.timeunit})
        self.app._kaldi_to_single_textdocument(pua_testcase, kaldi_view, d)
        self.assertEqual(2, len(m.views))
        self.assertEqual(1, len(list(kaldi_view.get_annotations(DocumentTypes.TextDocument))))
        self.assertEqual(10, len(list(kaldi_view.get_annotations(Uri.TOKEN))))
        self.assertEqual(10, len(list(kaldi_view.get_annotations(AnnotationTypes.TimeFrame))))
        self.assertEqual(11, len(list(kaldi_view.get_annotations(AnnotationTypes.Alignment))))  # 10 token-tf + 1 td-ad

    def test__kaldi_to_segmented_textdocument(self):
        pua_testcase = self.generate_pua_testcase()
        
        m = Mmif(validate=False)
        
        self.generate_audiodocument_view(m)

        seg_view = m.new_view()
        seg_view.new_contain(AnnotationTypes.TimeFrame, **{'timeUnit': self.app.timeunit})

        # pua_testcase is 20 second. 
        # Using two segments, sum(original speech segments) must be 19 second (+1 silence gap)
        seg1 = seg_view.new_annotation(AnnotationTypes.TimeFrame)
        seg1.add_property('frameType', 'speech')
        seg1.add_property('start', 0)
        seg1.add_property('end', 10 * self.app.timeunit_conv[self.app.timeunit])
        seg2 = seg_view.new_annotation(AnnotationTypes.TimeFrame)
        seg2.add_property('frameType', 'speech')
        seg2.add_property('start', 20 * self.app.timeunit_conv[self.app.timeunit])
        seg2.add_property('end', 29 * self.app.timeunit_conv[self.app.timeunit])

        kaldi_view = m.new_view()
        kaldi_view.new_contain(AnnotationTypes.TimeFrame, **{'timeUnit': self.app.timeunit})
        self.app._kaldi_to_segmented_textdocument(pua_testcase, kaldi_view, seg_view)
        self.assertEqual(3, len(m.views))  # audiodoc, speechsegment, kaldi
        self.assertEqual(2, len(list(kaldi_view.get_annotations(DocumentTypes.TextDocument))))  # 2 speech segments
        # only 5th('6', 10-11sec) token ignores (fall under mid-gap)
        self.assertEqual(9, len(list(kaldi_view.get_annotations(Uri.TOKEN))))  
        self.assertEqual(9, len(list(kaldi_view.get_annotations(AnnotationTypes.TimeFrame))))
        self.assertEqual(10, len(list(kaldi_view.get_annotations(AnnotationTypes.Alignment))))  # 9 token-tf + 1 td-ad
    
    def test__patchwork_audiofile(self):
        self.app._patchwork_audiofile('in.wav', 'out.wav',
                                      [(1, 3), (5, 7), (9.1, 10.13), (12, 15)], dryrun=True)

if __name__ == '__main__':
    unittest.main()


