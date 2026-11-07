import unittest
from src.common import parse_answer
class T(unittest.TestCase):
 def test_format(self):
  s="Причина. Это объяснение.\nЧто сделать:\n1. Проверить.\n2. Вызвать мастера.\n\nСрочность: скоро\nКто: специалист\nКатегория: техника\nСложность: 4/10"
  self.assertTrue(parse_answer(s)["format_valid"])
 def test_order(self):
  s="Причина.\nЧто сделать:\n1. Проверить.\n2. Вызвать мастера.\n\nКто: сам\nСрочность: скоро\nКатегория: техника\nСложность: 4/10"
  self.assertFalse(parse_answer(s)["format_valid"])
 def test_word_limit(self):
  s=("слово "*151)+"\nЧто сделать:\n1. A\n2. B\n\nСрочность: скоро\nКто: сам\nКатегория: техника\nСложность: 4/10"
  self.assertFalse(parse_answer(s)["format_valid"])
if __name__=="__main__":unittest.main()
