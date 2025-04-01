# Notation
$L$: number of layers, the output layer

$l$: $l^{th}$ layer

$m$: number of examples

$n_x$ = $n^{[0]}$: number of features

$n_y$ = $n^{[L]}$: number of classes

$f$: activation function

$f'$: derivative of activation function

---

|| row-wise | column-wise |   
| :------:| :------: | :------------------------: |
|shapes|x.shape: (m, n_x)<br>y.shape: (m, n_y)<br>$w$.shape: (n_in, n_out)<br>$b$.shape: (1, n_out)|x.shape: (n_x, m)<br>y.shape: (n_y, m)<br>$w$.shape: (n_out, n_in)<br>$b$.shape: (n_out, 1)|
|forward|$$Z^{[l]} = A^{[l - 1]} \cdot  W^{[l]} + b^{[l]}$$ $$A^{[l]} = f(Z^{[l]})$$|$$Z^{[l]} = W^{[l]} \cdot A^{[l - 1]}   + b^{[l]}$$ $$A^{[l]} = f(Z^{[l]})$$|
|gradient|for output layer, <br> $$dZ^{[L]} = A^{[L]} - y$$ <br> for hidden layers, <br> $$dZ^{[l - 1]} = dZ^{[l]} \cdot W^{[l]^T}  \times f^{'}(Z^{[l - 1]}), \quad dA^{[l]} = dZ^{[l]} \cdot W^{[l]^T}$$ $$dZ^{[l]} = dA^{[l]}  \times f^{'}(Z^{[l]})$$ | for output layer, <br> $$dZ^{[L]} = A^{[L]} - y$$ <br> for hidden layers, <br> $$dZ^{[l - 1]} =  W^{[l]^T} \cdot dZ^{[l]}  \times f^{'}(Z^{[l - 1]}), \quad dA^{[l]} = W^{[l]^T} \cdot dZ^{[l]}$$ $$dZ^{[l]} = dA^{[l]}  \times f^{'}(Z^{[l]})$$|
|update|for output layer, $$\frac {\partial {J}} {\partial {W}} = \frac {1} {m}   X^T \cdot (A - y),  \quad dW^{[l]} = \frac {1} {m} A^{[l - 1]^T} \cdot  dZ^{[l]}$$ $$\frac {\partial {J}} {\partial {b}} = \frac {1} {m} \sum(A - y), \quad db^{[l]} = \frac {1} {m} \sum dZ^{[l]}$$| for output layer, $$\frac {\partial {J}} {\partial {W}} = \frac {1} {m} (A - y) \cdot X^T, \quad dW^{[l]} = \frac {1} {m} dZ^{[l]}  \cdot  A^{[l - 1]^T} $$ $$\frac {\partial {J}} {\partial {b}} = \frac {1} {m} \sum(A - y), \quad db^{[l]} = \frac {1} {m} \sum dZ^{[l]}$$|
