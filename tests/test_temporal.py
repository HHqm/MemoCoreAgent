import unittest
from datetime import date

from memocore.datasets.locomo import load_locomo
from memocore.temporal import (
    TimeWindow,
    anchor_conversation,
    anchor_text,
    parse_session_date,
    question_window,
    time_note,
    turn_spans,
)
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


class QuestionWindowTest(unittest.TestCase):
    def test_days_months_and_bare_months(self):
        self.assertEqual(question_window("Who did Maria have dinner with on May 3, 2023?"),
                         TimeWindow(date(2023, 4, 26), date(2023, 5, 3)))
        self.assertEqual(question_window("What did she start, as mentioned on 3 June, 2023?"),
                         TimeWindow(date(2023, 5, 27), date(2023, 6, 3)))
        self.assertEqual(question_window("What did Maria donate in December 2023?"),
                         TimeWindow(date(2023, 12, 1), date(2023, 12, 31)))
        self.assertEqual(question_window("When did Melanie go camping in June?"), TimeWindow(month=6))

    def test_no_window_without_an_explicit_period(self):
        for question in ("What may Caroline do next?", "When did John meet Sam?", "How often did she run in 2023?"):
            self.assertIsNone(question_window(question))

    def test_turn_spans_cover_the_session_day_and_resolved_times(self):
        spans = turn_spans(date(2023, 7, 3), "time: last week = the week before 3 July 2023; next month = August 2023")
        self.assertIn((date(2023, 6, 26), date(2023, 7, 3)), spans)
        self.assertIn((date(2023, 8, 1), date(2023, 8, 31)), spans)
        self.assertNotIn((date(2023, 7, 1), date(2023, 7, 31)), spans)  # July only as part of a day
        self.assertTrue(TimeWindow(month=8).overlaps(date(2023, 8, 1), date(2023, 8, 31)))
        self.assertFalse(TimeWindow(date(2023, 9, 1), date(2023, 9, 30)).overlaps(date(2023, 8, 1), date(2023, 8, 31)))
