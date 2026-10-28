Hardware: training: CPU: x86_64 | torch 2.14.1+cu130 | inference: CPU

### Table 2
| Model | MSE1 | MSE2 | MSE3 | MSE4 | MSE5 | E(5) | Params |
|---|---|---|---|---|---|---|---|
| A | 0.0456 (seed 0 only) | 0.0446 (seed 0 only) | 0.0446 (seed 0 only) | 0.0445 (seed 0 only) | 0.0444 (seed 0 only) | 0.0447 (seed 0 only) | 480,784 |
| Blank | 0.0309 | 0.0310 | 0.0311 | 0.0310 | 0.0308 | 0.0310 | 0 |
| Copy-last | 0.0274 | 0.0460 | 0.0551 | 0.0583 | 0.0595 | 0.0493 | 0 |

### Table 3
| Model | E(5) | C_M | E_M | C_R | E_R |
|---|---|---|---|---|---|
| A | 0.0447 (seed 0 only) | 0.00226 (seed 0 only) | 0.0447 (seed 0 only) | 0.00261 (seed 0 only) | 0.0447 (seed 0 only) |
| Blank | 0.0310 | 0 | 0.0310 | 0 | 0.0310 |
| Copy-last | 0.0493 | 0 | 0.0493 | 0 | 0.0493 |

### Table 4
| Model | E(2) 1× | E(2) 2× | E(2) 3× | E(5) 1× | E(5) 2× |
|---|---|---|---|---|---|
| A | 0.0451 (seed 0 only) | 0.0450 (seed 0 only) | 0.0451 (seed 0 only) | 0.0447 (seed 0 only) | 0.0447 (seed 0 only) |
| Copy-last | 0.0367 | 0.0522 | 0.0578 | 0.0493 | 0.0568 |

### Table 5
| Model | E(5) 1× | C_M | E_M | C_R | E_R | E(2) 1× | E(2) 2× | E(2) 3× | Params | Train s/epoch | Inference s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0447 (seed 0 only) | 0.00226 (seed 0 only) | 0.0447 (seed 0 only) | 0.00261 (seed 0 only) | 0.0447 (seed 0 only) | 0.0451 (seed 0 only) | 0.0450 (seed 0 only) | 0.0451 (seed 0 only) | 480,784 | 20.6 (seed 0 only) | 3.47 (seed 0 only) |
| B | 0.0451 (seed 0 only) | 0.00240 (seed 0 only) | 0.0451 (seed 0 only) | 0.00276 (seed 0 only) | 0.0451 (seed 0 only) | 0.0454 (seed 0 only) | 0.0453 (seed 0 only) | 0.0455 (seed 0 only) | 480,784 | 15.3 (seed 0 only) | 3.72 (seed 0 only) |
| C | 0.0457 (seed 0 only) | 0.00260 (seed 0 only) | 0.0457 (seed 0 only) | 0.00365 (seed 0 only) | 0.0457 (seed 0 only) | 0.0460 (seed 0 only) | 0.0458 (seed 0 only) | 0.0460 (seed 0 only) | 480,784 | 24.4 (seed 0 only) | 2.20 (seed 0 only) |
| D | 0.0461 (seed 0 only) | 0.00258 (seed 0 only) | 0.0461 (seed 0 only) | 0.00341 (seed 0 only) | 0.0461 (seed 0 only) | 0.0464 (seed 0 only) | 0.0463 (seed 0 only) | 0.0464 (seed 0 only) | 480,784 | 15.3 (seed 0 only) | 2.14 (seed 0 only) |

### Table 6
| Model | E(5) 1× | C_M | E_M | C_R | E_R | E(2) 1× | E(2) 2× | E(2) 3× | Params | Train s/epoch | Inference s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sym B | 0.0440 (seed 0 only) | 1.02e-16 (seed 0 only) | 0.0440 (seed 0 only) | 8.75e-17 (seed 0 only) | 0.0440 (seed 0 only) | 0.0444 (seed 0 only) | 0.0443 (seed 0 only) | 0.0444 (seed 0 only) | 480,784 | 15.3 (seed 0 only) | 18.3 (seed 0 only) |
| sym D | 0.0448 (seed 0 only) | 1.13e-16 (seed 0 only) | 0.0448 (seed 0 only) | 9.01e-17 (seed 0 only) | 0.0448 (seed 0 only) | 0.0452 (seed 0 only) | 0.0451 (seed 0 only) | 0.0452 (seed 0 only) | 480,784 | 15.3 (seed 0 only) | 17.6 (seed 0 only) |
(missing: sym D+B)
