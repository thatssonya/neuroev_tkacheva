# neuroev_tkacheva

Работа алгоритма осуществляется в методе train класса ESPAlgorithm. Запускается цикл согласно количеству поколений (generations_count). В цикле вызывается метод check_fitness. В методе check_fitness запускается цикл, который выполняется до тех пор, пока каждый нейрон в каждой из подпопуляций не поучаствует в работе нейронной сети. Для данной проверки используется метод is_trials_completed класса NeuronPopulation. В цикле происходит выборка нейронов - случайно, по одному из каждой подпопуляции. Выбранные нейроны будут участвовать в работе нейронной сети, поэтому для них увеличивается счётчик количества попыток. Из выбранных нейронов создаётся нейронная сеть (объект NeuralNetwork). Для данной нейронной сети делается прямой проход по каждому набору входных данных (вызов вспомогательного метода forward_train). Для каждого набора данных вычисляется среднеквадратичная ошибка и возвращается среднее значение ошибок. Данное среднее значение будет являться кумулятивной приспособленностью выбранных нейронов. После того, как каждый нейрон поучавствует в работе нейронной сети, метод check_fitness завершает свою работу, возвращая количество попыток, которое потребовалось для того, чтобы каждый нейрон поучаствовал в работе сети. Получив значение кумулятивной приспособленности, происходит обновление средней приспособленности вызовом метода fit_avg_fitness класса NeuronPopulation. Затем делается проверка на вырождение подпопуляций вызовом метода check_degeneration класса NeuronPopulation. Далее, для каждой из подпопуляций делается скрещивание (1/4 часть наиболее приспособленных нейронов) вызовом метода crossover класса NeuronPopulation. Далее, для каждой из подпопуляций делается мутация вызовом метода mutation класса NeuronPopulation. В конце итерации происходит сброс счётчика попыток у каждого из нейронов. Данные шаги выполняются заданное количество раз.
