from app import *
import unittest
class Test(unittest.TestCase):
    def test_metrics(self):
        m=calculate(read_sales('data/cafe.csv')); self.assertEqual(m['revenue'],42000); self.assertEqual(m['profit'],18000); self.assertEqual(m['risks'],['Кофе'])
    def test_feedback(self):
        s=run_multi(initial('data/cafe.csv','test'),fixture_model,True); self.assertTrue(s['approved']); self.assertEqual(s['attempts'],2)
    def test_limit(self):
        s=run_multi(initial('data/cafe.csv','test'),lambda p:'bad',True); self.assertFalse(s['approved']); self.assertEqual(s['attempts'],3)
    def test_baseline(self): self.assertTrue(run_single(initial('data/cafe.csv','test'),fixture_model)['feedback'])
if __name__=='__main__': unittest.main()
