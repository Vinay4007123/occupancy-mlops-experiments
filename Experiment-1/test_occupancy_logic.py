import unittest

from occupancy_logic import is_room_occupied


class TestOccupancyLogic(unittest.TestCase):

    def test_empty_room(self):
        self.assertFalse(is_room_occupied(0))

    def test_one_person(self):
        self.assertTrue(is_room_occupied(1))

    def test_multiple_people(self):
        self.assertTrue(is_room_occupied(3))


if __name__ == "__main__":
    unittest.main()
