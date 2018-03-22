from .each import EachSample, EachToken


lower_case = lambda s: s.lower()

LowerCaseEachSample = lambda: EachSample(lower_case)

LowerCaseEachToken = lambda: EachToken(lower_case)
