"""Meaningful checks of sampling, temporal masking, rules and paired inference."""
import itertools
import unittest
import numpy as np
import pandas as pd
from traffic_impact.experiment import (FeatureBuilder, raw_features, OneRule, Model, RankedMetrics,
                        bootstrap_differences, score, POI, WNUM, WCAT, LIGHT, FORBIDDEN)
from traffic_impact.prepare import priorities, event_keys


def fixture():
    n = 400
    t = pd.date_range("2019-01-01", periods=n, freq="h")
    d = pd.DataFrame({"_t":t,"Start_Time":t.astype(str),"End_Time":(t+pd.Timedelta(hours=1)).astype(str),
        "Weather_Timestamp":(t-pd.Timedelta(minutes=30)).astype(str),
        "ID":[f"A-{i}" for i in range(n)],"Source":"Source2","Description":"accident",
        "Distance(mi)":.1,"Severity":2,"Start_Lat":np.linspace(30,35,n),"Start_Lng":-100.,
        "State":np.where(np.arange(n)%2,"TX","OH"),"Timezone":"US/Central"})
    for c in POI:
        d[c] = np.arange(n)%2 == 0 if c == "Traffic_Signal" else False
    for c in WNUM:
        d[c] = np.arange(n,dtype=float)
    for c in WCAT:
        d[c] = "NE" if c == "Wind_Direction" else "Clear"
    for c in LIGHT:
        d[c] = np.where(np.arange(n)%2,"Night","Day")
    d["_priority"] = np.arange(n,dtype=np.uint64)
    return d


class StudyChecks(unittest.TestCase):
    def test_weather_boundary_and_all_weather_fields(self):
        d = fixture()
        d.loc[0,"Weather_Timestamp"] = str(d.loc[0,"_t"]+pd.Timedelta(seconds=1))
        d.loc[1,"Weather_Timestamp"] = str(d.loc[1,"_t"]-pd.Timedelta(minutes=121))
        d.loc[2,"Weather_Timestamp"] = str(d.loc[2,"_t"]-pd.Timedelta(minutes=120))
        x,_,eligible = raw_features(d)
        self.assertFalse(eligible.iloc[0]); self.assertFalse(eligible.iloc[1]); self.assertTrue(eligible.iloc[2])
        self.assertTrue(x.loc[[0,1],WNUM+WCAT].isna().all().all())

    def test_train_categories_and_unknown_values(self):
        d = fixture()
        d["Wind_Direction"] = np.where(np.arange(len(d))%2,"NE","E")
        builder = FeatureBuilder().fit(d)
        v = d.copy(); v["State"] = "UNSEEN"; v["Wind_Direction"] = "SW"
        transformed = builder.transform(v)
        self.assertTrue(transformed.State.eq("Other").all())
        self.assertEqual(builder.vocab["State"],["OH","TX"])
        self.assertTrue(transformed.Wind_Direction.eq("Other").all())
        self.assertFalse(set(builder.columns) & FORBIDDEN)
        self.assertIn("Turning_Loop",builder.dropped)

    def test_priority_is_order_independent_and_label_independent(self):
        d = fixture()
        before = event_keys(d)
        d.Severity = 4
        np.testing.assert_array_equal(event_keys(d),before)
        np.testing.assert_array_equal(priorities(before[::-1]),priorities(before)[::-1])
        self.assertGreater(len(np.unique(priorities(before))),390)

    def test_threshold_finds_both_directions(self):
        x = pd.Series(np.arange(400,dtype=float))
        for y in [(x>100).to_numpy().astype(int),(x<300).to_numpy().astype(int)]:
            rule = OneRule().fit(x,y,"x")
            self.assertEqual(score(y,rule.predict(x),np.arange(len(y)))["ap"],1.)

    def test_unseen_category_falls_back_to_train_prior(self):
        x = pd.Series(pd.Categorical(["a"]*100+["b"]*100))
        y = np.r_[np.zeros(100),np.ones(100)]
        rule = OneRule().fit(x,y,"x")
        self.assertEqual(rule.predict(pd.Series(pd.Categorical(["unknown"])))[0],.5)

    def test_top_ten_lift_compares_precision_to_prevalence(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        p = np.linspace(0.99, 0.1, len(y))
        metrics = score(y, p, np.arange(10))
        self.assertEqual(metrics["precision10"], 1.0)
        self.assertEqual(metrics["prevalence"], 0.3)
        self.assertAlmostEqual(metrics["lift10"], 1 / 0.3)

    def test_weighted_metrics_match_explicit_repetition_with_ties(self):
        y=np.array([1,0,1,0,0,1,0,1,0,1,0,0])
        p=np.array([.5,.5,.2,.2,.2,.9,.9,.8,.8,.3,.3,.3])
        tie=np.arange(len(y))
        weights=np.array([2,0,1,3,1,2,0,4,1,0,1,2])
        idx=np.repeat(np.arange(len(y)),weights)
        actual=RankedMetrics(y,p,tie).weighted(weights)
        expected=score(y[idx],p[idx],tie[idx])
        np.testing.assert_allclose(actual,[expected["ap"],expected["recall10"]])

    def test_identical_predictions_have_zero_paired_interval(self):
        d=fixture()
        y=(np.arange(len(d))%3==0).astype(int)
        p=np.linspace(0,1,len(d))
        rows=bootstrap_differences(d,y,{"hgb_full":p,"same":p},B=20)
        for key in ["delta_ap","ap_lower","ap_upper","delta_recall10","recall10_lower","recall10_upper"]:
            self.assertEqual(rows[0][key],0.)

    def test_simple_models_handle_unseen_categories(self):
        d=fixture(); builder=FeatureBuilder().fit(d); x=builder.transform(d)
        y=(np.arange(len(d))%3==0).astype(int)
        v=d.iloc[:10].copy(); v["State"]="ZZ"; z=builder.transform(v)
        for kind in ["tree","logistic","hgb"]:
            model=Model(kind,["Start_Lat","State"],{"max_iter":10} if kind=="hgb" else {}).fit(x,y)
            p=model.predict(z)
            self.assertTrue(np.isfinite(p).all()); self.assertTrue(((p>=0)&(p<=1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
