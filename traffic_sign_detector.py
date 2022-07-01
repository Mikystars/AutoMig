import platform
import sys
import getopt
from typing import List, NamedTuple
import json
from tflite_support import metadata

import cv2
import tensorflow as tf
import numpy as np

import NanoCam as nano
import serial2arduino as ardu

_SHOW_IMAGE = True

# El interprete
Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate

# Opciones posibles a asignar al detector de objetos
class ObjectDetectorOptions(NamedTuple):
  """Config del detector de objetos."""

  # Probar esto en el Colab
  enable_edgetpu: bool = False
  """Activar la ejecución en EdgeTPU."""

  label_allow_list: List[str] = None
  """Etiquetas permitidas"""

  label_deny_list: List[str] = None
  """Etiquetas prohibidas"""

  max_results: int = -1
  """Numero de detecciones maxima a mostrar"""

  num_threads: int = 4
  """Hilos de CPU a usar."""

  score_threshold: float = 0.0
  """El minimo de detección a devolver por el detector"""


class Rect(NamedTuple):
  """Para dibujar rectangulo"""
  left: float
  top: float
  right: float
  bottom: float


class Category(NamedTuple):
  """Resultado de la clasificacion"""
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  """El objeto de tectado con el ObjectDetector."""
  bounding_box: Rect
  categories: List[Category]


def edgetpu_lib_name():
  """Devuelve la librería de edgeTPU en el sistema.
     La Jetson Nano no tiene TPUs, lo dejo para cuando use Google Colab
  """
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)


class ObjectDetector:
  """El contenedor para las detecciones de TFLite"""

  _OUTPUT_LOCATION_NAME = 'location'
  _OUTPUT_CATEGORY_NAME = 'category'
  _OUTPUT_SCORE_NAME = 'score'
  _OUTPUT_NUMBER_NAME = 'number of detections'

  def __init__(
      self,
      model_path: str,
      options: ObjectDetectorOptions = ObjectDetectorOptions()
  ) -> None:
    """Inicializar el objeto de deteccion de TFLite.
    Args:
        model_path: Ruta al archivo .tflite.
        options: La configuración creada con ObjectDetectorOptions. (Opcional)
    Errores:
        ValueError: El modelo tfLite no es válido.
        OSError: El sistema operativo no es soportado por EdgeTPU.
    """

    # Cargar los metadatos
    displayer = metadata.MetadataDisplayer.with_model_file(model_path)

    # Guardar los metadatos del modelo
    model_metadata = json.loads(displayer.get_metadata_json())
    process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']
    mean = 0.0
    std = 1.0
    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    self._mean = mean
    self._std = std

    # Obtener lista de etiquetas de los metadatos
    # TODO: Arreglar esto. La librería metadata no funciona. En Colab y en el PC va bien.
    #file_name = displayer.get_packed_associated_file_list()[0]
    #label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    #label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))

    # Etiquetas asignadas a mano porque la librería metadata no funciona
    self._label_list = ['stop', 'turn_right']

    # Inicializar modelo tfLite.
    # En la Jetson esto no sirve, en el Colab si.
    if options.enable_edgetpu:
      if edgetpu_lib_name() is None:
        raise OSError("The current OS isn't supported by Coral EdgeTPU.")
      interpreter = Interpreter(
          model_path=model_path,
          experimental_delegates=[load_delegate(edgetpu_lib_name())],
          num_threads=options.num_threads)
    else:
      interpreter = Interpreter(
          model_path=model_path, num_threads=options.num_threads)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    #Ordenar las salidas del modelo
    sorted_output_indices = sorted(
        [output['index'] for output in interpreter.get_output_details()])
    self._output_indices = {
        self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
        self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
        self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
        self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
    }

    self._input_size = input_detail['shape'][2], input_detail['shape'][1]
    self._is_quantized_input = input_detail['dtype'] == np.uint8
    self._interpreter = interpreter
    self._options = options

  def detect(self, input_image: np.ndarray) -> List[Detection]:
    """Realizar detección en una imagen.
    Args:
        input_image: Imagen [height, width, 3] RGB. Alto y ancho puede ser
        cualquiera, se hace resize después.
    """
    image_height, image_width, _ = input_image.shape

    input_tensor = self._preprocess(input_image)

    self._set_input_tensor(input_tensor)
    self._interpreter.invoke()

    # Obtener detalles
    boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
    classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
    scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
    count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

    return self._postprocess(boxes, classes, scores, count, image_width,
                             image_height)

  def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
    """Preprocesar la imgen para wue funcione con el modelo."""

    # Resize al tamaño correcto
    input_tensor = cv2.resize(input_image, self._input_size)

    # Normalizar la entrada si no está cuantizada
    if not self._is_quantized_input:
      input_tensor = (np.float32(input_tensor) - self._mean) / self._std

    # Añadir una dimensión
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    #Devuelve el tensor de entrada
    return input_tensor

  def _set_input_tensor(self, image):
    """Coloca el input tensor."""
    tensor_index = self._interpreter.get_input_details()[0]['index']
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def _get_output_tensor(self, name):
    """Devuelve el output tensor en el indice dado."""
    output_index = self._output_indices[name]
    tensor = np.squeeze(self._interpreter.get_tensor(output_index))
    return tensor

  def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                   scores: np.ndarray, count: int, image_width: int,
                   image_height: int) -> List[Detection]:
    """Postprocesado de la salida del modelo en una lista de detección de objetos.
       Returns:
        Una lista con las detecciones
    """
    results = []

    for i in range(count):
      if scores[i] >= self._options.score_threshold:
        y_min, x_min, y_max, x_max = boxes[i]
        bounding_box = Rect(
            top=int(y_min * image_height),
            left=int(x_min * image_width),
            bottom=int(y_max * image_height),
            right=int(x_max * image_width))
        class_id = int(classes[i])
        category = Category(
            score=scores[i],
            label=self._label_list[class_id],
            index=class_id)
        result = Detection(bounding_box=bounding_box, categories=[category])
        results.append(result)

    # Ordenar los resultados por score, ascendente
    sorted_results = sorted(
        results,
        key=lambda detection: detection.categories[0].score,
        reverse=True)

    # Filtrar detecciones en la deny list
    # Esto se indica en el objeto de opciones
    filtered_results = sorted_results
    if self._options.label_deny_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label not in self.
              _options.label_deny_list, filtered_results))

    # Mantener solo detecciones en la allow list
    # Se indica en el objeto de opciones
    if self._options.label_allow_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label in self._options.
              label_allow_list, filtered_results))

    # Devolver solo un máximo de detecciones
    # Se indica en el objeto de opciones
    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 255, 0)  # Texto de deteccion en verde


def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
  """Dibuja las cajas de deteccion y devuelve la imagen.
  Args:
    image: La imagen RGB.
    detections: La lista de detecciones a pintar sobre la imagen.
  Returns:
    Imagen con las cajas pintadas.
  """
  for detection in detections:
    # Dibuja la caja
    start_point = detection.bounding_box.left, detection.bounding_box.top
    end_point = detection.bounding_box.right, detection.bounding_box.bottom
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Dibuja etiqueta y puntuación
    category = detection.categories[0]
    class_name = category.label
    probability = round(category.score, 2)
    result_text = class_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + detection.bounding_box.left,
                     _MARGIN + _ROW_SIZE + detection.bounding_box.top)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


def test_image(detector, image):
    cap = cv2.imread(image)
    
    #cap = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    image_np = np.asarray(cap)

    detections = detector.detect(image_np)

    # Draw keypoints and edges on input image
    image_np = visualize(image_np, detections)

    cv2.imshow('Detecciones', image_np)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video(detector, video_file):
    cap = cv2.VideoCapture(video_file)

    for i in range(3):
        _, image = cap.read()

    try:
        i = 0
        while cap.isOpened():
            ret, image = cap.read()
            if ret:
                print('frame %s' % i )
                
                image_np = np.asarray(image)
                
                detections = detector.detect(image_np)

                # Draw keypoints and edges on input image
                image_np = visualize(image_np, detections)

                if record:
                    video_overlay.write(image_np)

                cv2.imshow('Detectando...', image_np)
                
                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def test_camera(detector, record=0):
    cap = nano.Camera(flip=0, width=200, height=200, fps=20)

    # El tipode video que voy a guardar. Codec: XVID
    video_type = cv2.VideoWriter_fourcc(*'XVID')
    
    # El ancho y alto del stream
    video_w = cap.width
    video_h = cap.height
    video_fps = cap.fps
    
    if record:
        # El overlay del video. Mismo ancho x alto que la captura y 20 fps.
        video_overlay = cv2.VideoWriter("video_out/%s_overlay.avi" % ('CSIcamera'), video_type, video_fps, (video_w, video_h))

    try:
        while 1:
            image = cap.read()
            image_np = np.asarray(image)


            # Run object detection estimation using the model.
            detections = detector.detect(image_np)

            for det in detections:
                label = det[1][0].label  
                if label == 'stop':
                    print('STOP DETECTADO')
                    ardu.throttle(30)

                if label == 'turn_right':
                    print('GIRO DERECHA DETECTADO')
                    ardu.turn(130)

            # Draw keypoints and edges on input image
            image_np = visualize(image_np, detections)

            if record:
                video_overlay.write(image_np)

            cv2.imshow('Detectando...', image_np)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        ardu.throttle(30)
        cap.release()
        if record:
            video_overlay.release()
        cv2.destroyAllWindows()

def main(test_type=2, file_name='test_image.jpg'):
    '''Funcion principal, realiza el seguimiento de carril en imagen, video o cámara.
    Args:
    test_type: 0 para camara, 1 para video, 2 para imagen.
    '''
    DETECTION_THRESHOLD = 0.5
    TFLITE_MODEL_PATH = "models/trafficDetection.tflite"
    
    
    # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=4,
        score_threshold=DETECTION_THRESHOLD,
    )
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)
    
    if test_type == 0:
        test_camera(detector)

    elif test_type == 1:
        test_video(detector, file_name)

    elif test_type == 2:
        test_image(detector, file_name)




if __name__ == '__main__':
    mode = 2
    file_path = 'test_image.jpg'

    try:
        options, remainder = getopt.gnu_getopt(
            sys.argv[1:],
            'm:f:h',
            ['mode=',
            'file=',
            'help'
            ])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)

    #print('OPTIONS   :', options) 

    for opt, arg in options:
        if opt in ('-m', '--mode'):
            print('Modo: ' + str(arg))
            mode = int(arg)

        if opt in ('-f', '--file'):
            print('Archivo: ' + str(arg))
            file_path = str(arg)

        if opt in ('-h', '--help'):
            print('''
            --- Ayuda para el detector de señales ---
            Argumentos:

            -m [0, 1, 2]:
            (0: Detectar en la camara, 
             1: Detectar en archivo de video,
             2: Detectar en archivo de imagen)

             -f [ruta_fichero]:
             (La direccion de fichero de video o imagen a tratar)

             -h:
             (Muestra esta información de ayuda)

             NOTA: Si no se indica modo ni fichero, por defecto se toma
             '-m 2 -f test_image.jpg' como argumentos.
            ''')
            sys.exit(0)


    main(mode, file_path)
