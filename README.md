# Multilayer Perceptron: Feed Forward Propagation

## Параметры инициализации

* `X` - Входной массив объектов размером 2 * 4
* `Y` - Выходные метки со значениями размера 1 * 4
* `nx` - Количество нейронов во входном слое
* `nh` - Количество нейронов в скрытом слое
* `ny` - Количество нейронов в выходном слое
* `W1` - Веса в первом слое нейронной сети размера (nh * nx)
* `b1` - Смещение в первом слое нейронной сети размером (nh * 1)
* `W2` - Веса во втором слое нейронной сети размером (ny * nh)
* `b2` - Смещение во втором слое нейронной сети размером (ny * 1)
* `learning_rate` - Скорость обучения инициализирована значением 0,1
* `epochs` - Эпохи инициализированы с 10000

## Вызов функции прямого распространения

### функция `forward_propagation`
Функция принимает входные данные `X`, веса `W` и `W2` на уровне 1 и 2 соответственно 
вместе сo смещением на соответствующих уровнях `b1` и `b2`.

* Размеры `X` 2 * 4 и `W1` 2 * 2. `np.dot()` вычисляет скалярное произведение входных данных `X` и весов `W1`.
К результату добавляется смещение, чтобы вычислить чистый выход на скрытом уровне `net_h` размера 2 * 4. 
Этот чистый выход на скрытом слое `net_h` преобразуется с использованием функции активации сигмоида для вычисления `out_h`.

* Выходные данные на скрытом слое `out_h` умножаются на веса слоя 2, `W2`, а затем добавляется смещение для вычисления 
чистого выхода `net_y` на выходном слое. Чистые выходные размеры составляют 1 * 4. Чистый вывод `net_y` активируется путем 
применения сигмоидной функции активации для вычисления `out_y`.

Ниже приведены размеры параметров, передаваемых в функцию:

      net_h(2 * 4) = W1(2 * 2) * X(2 * 4) + b1 (2 * 1) 
      out_h(2 * 4) = sigmoid(net_h)
      net_y(1 * 4) = W2(1 * 2) * out_h (2 * 4) + b2 (1, 1)
      out_y(1 * 4) = sigmoid(net_y)
