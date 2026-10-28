| Check | Expected | Yours |
|---|---|---|
| (a) max|M(M(x)) − x| | 0 | 0 (torch.equal: True) |
| (b) max|R⁴(x) − x| | 0 | 0 (torch.equal: True) |
| (c) max|MRM(x) − R⁻¹(x)| | 0 | 0 (torch.equal: True) |
| (d) bright pixel of M(p) (frame, row, column) | (7, 10, 43) | (7, 10, 43) |
| (e) bright pixel of R(p) (frame, row, column) | (7, 43, 10) | (7, 43, 10) |
| (f) 1× input / target frames | 1–5 / 6–10 | 1–5 / 6–10 |
| (g) 2× input / target frames | 1, 3, 5, 7, 9 / 11, 13, 15, 17, 19 | 1, 3, 5, 7, 9 / 11, 13, 15, 17, 19 |
| (h) 3× input / target frames | 1, 4, 7, 10, 13 / 16, 19 | 1, 4, 7, 10, 13 / 16, 19 |
