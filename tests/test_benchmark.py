import unittest

from benchmark import create_parser


class BenchmarkDefaultsTest(unittest.TestCase):
    def test_all_uses_tutorial_sized_defaults(self):
        arguments = create_parser().parse_args(["--all"])

        self.assertEqual((arguments.m, arguments.n, arguments.k), (4096, 4096, 4096))
        self.assertEqual(arguments.warmup, 5)
        self.assertEqual(arguments.repeat, 50)

    def test_explicit_values_override_defaults(self):
        arguments = create_parser().parse_args(
            [
                "--kernel",
                "10",
                "--m",
                "512",
                "--n",
                "768",
                "--k",
                "256",
                "--warmup",
                "2",
                "--repeat",
                "7",
            ]
        )

        self.assertEqual((arguments.m, arguments.n, arguments.k), (512, 768, 256))
        self.assertEqual(arguments.warmup, 2)
        self.assertEqual(arguments.repeat, 7)


if __name__ == "__main__":
    unittest.main()
