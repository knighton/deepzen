from .each import EachSample, EachToken


lower = lambda s: s.lower()

LowerEachSample = lambda: EachSample(lower)

LowerEachToken = lambda: EachToken(lower)
