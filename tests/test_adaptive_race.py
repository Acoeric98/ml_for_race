import math
import unittest
import numpy as np
from adaptive_race import Race, TrackMemory, opponent_observation, resolve_collisions
from racing import Car, left_right_edges


def circle():
    angles=np.linspace(0,2*math.pi,80,endpoint=False)
    center=np.column_stack([50*np.cos(angles),50*np.sin(angles)])
    widths=np.full(80,12.)
    left,right=left_right_edges(center,widths)
    return dict(center=center,widths=widths,left=left,right=right)


class AdaptiveTests(unittest.TestCase):
    def test_opponents_are_relative_and_padded(self):
        a=Car((0,0)); b=Car((10,2)); b.v=5
        obs=opponent_observation(a,[a,b,Car((500,0))])
        np.testing.assert_allclose(obs,[.125,.025,5/45,1,0,0,0,0])

    def test_collision_separates_and_slows_closing_cars(self):
        a=Car((0,0)); a.v=10
        b=Car((2,0)); b.v=0
        self.assertEqual(resolve_collisions([a,b]),[True,True])
        self.assertGreaterEqual(b.x-a.x,2.6)
        self.assertAlmostEqual(a.v,5)
        self.assertAlmostEqual(b.v,5)

    def test_memory_laps_reverse_and_isolation(self):
        t=circle(); memory=TrackMemory(t); other=TrackMemory(t)
        memory.observe(0,10,0,False,0)
        for i in range(1,161):
            memory.observe(i%80,10,0,False,i)
        self.assertEqual(memory.laps,2)
        self.assertEqual(len(memory.lap_times),2)
        self.assertGreater(memory.target(0),other.target(0))
        before=memory.limits[0]
        memory.observe(0,8,2,False,162)
        self.assertLess(memory.limits[0],before)
        self.assertEqual(other.laps,0)
        reverse=TrackMemory(t)
        for i in range(161):
            reverse.observe((-i)%80,10,0,False,i)
        self.assertEqual(reverse.laps,0)

    def test_controller_completes_two_survey_laps_then_races(self):
        race=Race(circle(),[None])
        for _ in range(3000):
            race.step()
            if race.memories[0].laps >= 3:
                break
        result=race.results()[0]
        self.assertGreaterEqual(result['laps'],3)
        self.assertEqual(result['off_time'],0)
        self.assertLess(result['lap_times'][2],result['lap_times'][0])

    def test_collision_toggle_and_grid(self):
        race=Race(circle(),[None,None],collisions=False)
        self.assertGreater(math.hypot(race.cars[0].x-race.cars[1].x,race.cars[0].y-race.cars[1].y),2.6)
        for c in race.cars:
            c.x,c.y=50,0
        race.step()
        self.assertEqual(sum(race.contacts),0)
        race.collisions=True
        race.step()
        self.assertGreater(sum(race.contacts),0)

    def test_high_speed_contact_is_not_skipped(self):
        race=Race(circle(),[None,None],True)
        a,b=race.cars
        a.x,a.y,a.heading,a.v=45,0,0,90
        b.x,b.y,b.heading,b.v=50,0,math.pi,90
        race.step()
        self.assertTrue(all(race.contacts > 0))
        self.assertLess(a.v,90)
        self.assertLess(b.v,90)

    def test_network_receives_opponents_and_adaptation(self):
        class Net:
            def activate(self,obs):
                self.obs=obs
                return [0,0,0]
        net=Net(); race=Race(circle(),[net,None],True)
        race.step()
        self.assertEqual(len(net.obs),36)
        self.assertEqual(net.obs[29],1)
        self.assertTrue(np.isfinite(net.obs).all())


if __name__=='__main__':
    unittest.main()
