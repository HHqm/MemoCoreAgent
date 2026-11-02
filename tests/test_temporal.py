import unittest
from datetime import date

from memocore.datasets.locomo import load_locomo
from memocore.temporal import anchor_conversation, anchor_text, parse_session_date, time_note
from tests.helpers import FIXTURE

THURSDAY = date(2023, 5, 25)


class TemporalAnchorTest(unittest.TestCase):
    def resolve(self, text, said_on=THURSDAY):
        return dict(anchor_text(text, said_on))

    def test_parse_session_date(self):
        self.assertEqual(parse_session_date("1:56 pm on 8 May, 2023"), date(2023, 5, 8))
        self.assertIsNone(parse_session_date("sometime"))

    def test_days(self):
        self.assertEqual(self.resolve("I went yesterday")["yesterday"], "24 May 2023")
        self.assertEqual(self.resolve("the day before yesterday")["the day before yesterday"], "23 May 2023")
        self.assertEqual(self.resolve("Tomorrow!")["Tomorrow"], "26 May 2023")
        self.assertEqual(self.resolve("two days ago")["two days ago"], "23 May 2023")

    def test_weeks_and_weekdays_match_locomo_answer_style(self):
        self.assertEqual(self.resolve("last week")["last week"], "the week before 25 May 2023")
        self.assertEqual(self.resolve("last weekend")["last weekend"], "the weekend before 25 May 2023")
        self.assertEqual(
            self.resolve("last Sunday")["last Sunday"], "the Sunday before 25 May 2023 (21 May 2023)"
        )
        # Said on a Thursday, "last Thursday" means a week earlier, not today.
        self.assertIn("(18 May 2023)", self.resolve("last Thursday")["last Thursday"])
        self.assertEqual(self.resolve("next Saturday")["next Saturday"], "27 May 2023")

    def test_months_years_and_seasons(self):
        self.assertEqual(self.resolve("last month")["last month"], "April 2023")
        self.assertEqual(self.resolve("next month", date(2023, 12, 3))["next month"], "January 2024")
        self.assertEqual(self.resolve("last year")["last year"], "2022")
        self.assertEqual(self.resolve("three years ago")["three years ago"], "2020")
        self.assertEqual(self.resolve("a few years ago")["a few years ago"], "a few years before 25 May 2023")
        self.assertEqual(self.resolve("last summer")["last summer"], "summer 2022")
        self.assertEqual(self.resolve("last summer", date(2023, 7, 10))["last summer"], "summer 2022")
        self.assertEqual(self.resolve("this summer")["this summer"], "summer 2023")

    def test_longest_phrase_wins_and_order_is_kept(self):
        anchors = anchor_text("Last weekend was fun, and the day before yesterday too", THURSDAY)
        self.assertEqual([expression for expression, _ in anchors], ["Last weekend", "the day before yesterday"])
        self.assertIsNone(time_note("No dates here.", THURSDAY))

    def test_conversation_keeps_raw_text_and_adds_notes(self):
        conversation = load_locomo(FIXTURE)[0].conversation
        anchored = anchor_conversation(conversation)
        adopted = anchored.sessions[0].turns[0]
        self.assertEqual(adopted.text, conversation.sessions[0].turns[0].text)
        self.assertEqual(adopted.time_note, "time: yesterday = 7 May 2023")
        self.assertIn("[time: yesterday = 7 May 2023]", adopted.render())
        moved = anchored.sessions[1].turns[0]
        self.assertEqual(moved.time_note, "time: last week = the week before 20 June 2023")
        self.assertIsNone(anchored.sessions[0].turns[1].time_note)


if __name__ == "__main__":
    unittest.main()
