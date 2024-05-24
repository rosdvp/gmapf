# G-MAPF - A* и Collision Avoidance для мульти-агентных систем на GPU в реальном времени
В репозитории содержится исходный код алгоритма, объединяющего в себе поиск пути и перемещение с учетом Collision Avoidance. Алгоритм расчитан на обработку свыше 2000 агентов с уникальными маршрутами в пределах одного кадра. Алгоритм исполняется полностью на GPU с использованием инструментария CUDA.

Проект является частью магистерской выпускной квалификационной работы по образовательной программе "Технологии разработки компьютерных игр" [Школы разработки видеоигр Университета ИТМО](https://itmo.games/).

## Установка
### Пререквизиты
* C++ 17
* [CUDA Toolkit 12.5](https://developer.nvidia.com/cuda-downloads) (установка Visual Studio с расширение CUDA не обязательна)

### Visual Studio
1) Откройте Configuration Properties своего Solution-а. 
2) В C++ -> General -> Additional Include Directories добавьте путь к папке gmapf/include.
3) Туда же добавьте путь к папке CudaToolkit/include.
4) В VC++ Directories -> Library Directories добавьте путь к папке gmapf/X64/Release.
5) Туда же добавьте путь к папке CudaToolkit/lib.
6) В Linker -> Input -> Additional Dependencies укажите следующие названия: gmapf.lib, cuda.lib, cudadevrt.lib, cudart_static.lib.
 
### Unreal Engine
В Unreal Engine проекте по умолчанию находится файл *.Build.cs.
Добавьте в него следующий код и отредактируйте пути так, чтобы они указывали на папки алгоритма и на директорию установки CUDA Toolkit.

```csharp
// *.Build.cs

var gmapfInclude = "<insert_path>/gmapf/include";
var gmapfLib     = "<insert_path>/gmapf/x64/Release/gmapf.lib";
PublicIncludePaths.Add(gmapfInclude);
PublicAdditionalLibraries.Add(gmapfLib);

// Optional, include embedded CDT for NavMesh
var cdtInclude = "<insert_path>/gmapf/CDT";
PublicIncludePaths.Add(cdtInclude);
 
var cudaInclude = "<insert_path>/CudaToolkit/include";
var cudaLib  = "<insert_path>/CudaToolkit/lib/x64";
PublicIncludePaths.Add(cudaInclude);
PublicAdditionalLibraries.Add(Path.Combine(cudaLib, "cuda.lib"));
PublicAdditionalLibraries.Add(Path.Combine(cudaLib, "cudadevrt.lib"));
PublicAdditionalLibraries.Add(Path.Combine(cudaLib, "cudart_static.lib"));
```

### Сборка из исходников
1) Установите Visual Studio с расширением CUDA.
2) Откройте gmapf.sln файл.
3) main.cpp содержит тестовый код для запуска.

## API
```cpp
#include "gmapf/GMAPF.h"

gmapf::Config config;       
config.AgentsMaxCount = 2048;           // Максимальное число агентов
config.AgentSpeed = 1;                  // Скорость перемещения агентов
config.AgentRadius = 2;                 // Радиус коллизии агентов
config.PathFinderParallelAgents = 2048; // Максимальное число обрабатываемых за 1 вызов алгоритма запросов на поиск пути
config.PathFinderThreadsPerAgents = 4;  // Количество потоков на каждого агента, минимум 2.
config.PathFinderQueueCapacity = 16;    // Вместимость каждой приоритетной очереди A* алгоритма на каждом потоке. 
config.PathFinderHeuristicK = 1;        // Множитель эвристики A* алгоритма.
config.IsProfiler = true;               // Включение режима профайлера (выключен по умолчанию).

gmapf::GMAPF gmapf;
gmapf.Init(config);

// Заполнение карты идет до добавления агентов и после инициализации.
gmapf.FillMapStart(100, 100);   // Начинаем заполнять карту размером 100х100
std::vector<gmapf::V2Float> wall;   // Задаем точки стены
wall.emplace_back(30, 40);
wall.emplace_back(30, 60);
wall.emplace_back(60, 40);
wall.emplace_back(60, 60);
gmapf.FillMapObstacle(wall);    // Добавляем стену на карту
...
gmapf.FillMapEnd();     // Заканчиваем собирать карту

// Добавление агентов идет после заполнения карты и до запуска алгоритма.
gmapf.AddAgent({0, 0});     // Добавляем агента на позицию (0, 0)
gmapf.SetAgentTargetPos(0, {100, 100}); // Задаем первому агенту цель

gmapf.ManualStart();    // Пред-аллоцируем ресурсы алгоритма (опционально)

while (...)
{
    gmapf.AsyncStep(deltaTime); // Запускаем шаг алгоритма
    // Между запуском и концом шага CPU может свободно использоваться для других задач
    gmapf.WaitStepEnd();        // Блокируем CPU до завершения шага алгоритма

    gmapf::V2Float agentNewPos = gmapf.GetAgentPos(0); // Получаем текущую позицию агента

    if (gmapf.GetAgentState(0) == EAgentState::Idle) // Получаем текущее состояние агента
        gmapf.SetAgentTargetPos(0, {0, 0}); // Назначать целевые позиции агентам можно в любое время, кроме выполнения шага алгоритма
}

gmapf.ProfilerDump(); // Выводим информацию о производительности

// Ресурсы алгоритма будут освобождены при вызове деструктора.
```

Пояснения:<br/>
1. Количество потоков на каждого агента подбирается эмпирически: чем сложнее граф, тем больший прирост скорости будет получен из дополнительных потоков. Возможные значения: 2, 4, 8, 16, 32.
2. Вместимость приоритетной очереди подбирается эмперически: чем сложнее граф, тем больше должна быть вместимость для успешного поиска.
3. Вместо встроенного CDT решения для генерации NavMesh-а возможно применение стороннего решения. Для этого нужно передать в метод FillMapManually проходимые узлы, указав для каждого три вершины и три индекса соседних узлов.


## Дополнительно
* Подробное описание алгоритма доступно по [ссылке](docs/VKR.pdf).
* [Пример](https://youtu.be/21h0ROmfy9I) работы алгоритма в Unreal Engine на 4000 агентах при пиковой нагрузке в 4.5ms (RTX 3060).
* Алгоритм использует [библиотеку CDT](https://github.com/artem-ogre/CDT) для генерации NavMesh-а.