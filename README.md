# MAZE HUNTER - Reinforcement Learning Environment

![alt text](https://github.com/alexanderplatas/RL-maze_hunter/blob/main/gameplay_example.gif?raw=true)

## Aprendizaje Reforzado Multiagente en un Entorno de Laberinto

Este repositorio contiene código para entrenar y evaluar agentes en un entorno de laberinto utilizando la biblioteca Stable Baselines3.

### Entorno

El entorno de laberinto simula un escenario de caza con agentes de presa y cazador. El laberinto está envuelto para facilitar el proceso de entrenamiento.

#### MazeWrapper

La clase `MazeWrapper` es un envoltorio personalizado para el entorno de OpenAI Gym. Modifica el espacio de observación para separar las observaciones de presa y cazador, facilitando el manejo durante el entrenamiento y la evaluación.

### Entrenamiento

Para entrenar agentes, utiliza los siguientes argumentos de línea de comandos:

```bash
python main.py train <tipo_agente>
```

- `<tipo_agente>` puede ser 'prey' o 'hunter'.

#### Ejemplo:

```bash
python main.py train prey
```

### Evaluación

Para evaluar agentes, utiliza el siguiente argumento de línea de comandos:

```bash
python main.py eval
```

- Esto evalúa los agentes preentrenados de presa y cazador en el entorno de laberinto.

### Uso

1. Clona el repositorio:

```bash
git clone https://github.com/tu_nombre_usuario/lab-rl.git
cd lab-rl
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Entrena a los agentes:

```bash
python main.py train prey
python main.py train hunter
```

4. Evalúa a los agentes:

```bash
python main.py eval
```

