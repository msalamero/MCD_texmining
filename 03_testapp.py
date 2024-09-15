import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Cargar los datos
#df = pd.read_csv('D:/Repos/MCD_Textmining/reviews_final2_3_clas.csv')
#df2 = pd.read_csv('D:/Repos/MCD_Textmining/stream1000.csv') 
df = pd.read_csv('reviews_final2_3_clas.csv')
df2 = pd.read_csv('stream1000.csv')  # Make sure this file is in the same directory

# Limpieza de datos
df['Fecha'] = df['Fecha;;;;'].str.replace(';;;;', '').str.strip()
df.drop(columns=['Fecha;;;;'], inplace=True)
df['clasificacion'] = df['clasificacion'].str.split(',').apply(lambda x: [item.strip() for item in x])
df['valoracion'] = df['valoracion'].str.split(',').apply(lambda x: [item.strip() for item in x])

# Análisis de sentimiento
def analizar_sentimiento(valoracion):
    positivo = sum(1 for v in valoracion if v == 'Positivo')
    negativo = sum(1 for v in valoracion if v == 'Negativo')
    neutro = sum(1 for v in valoracion if v == 'Neutro')
    
    if positivo > max(negativo, neutro):
        return 'Positivo'
    elif negativo > max(positivo, neutro):
        return 'Negativo'
    else:
        return 'Neutro'

df['Sentimiento'] = df['valoracion'].apply(analizar_sentimiento)

# Crear menú de navegación en la barra lateral
st.sidebar.title("Navegación")
menu = st.sidebar.selectbox('Selecciona una página:', 
                            ['Inicio', 'Web Scraping', 'Limpieza de Datos', 'Análisis de Sentimientos', 'Visualización de Resultados'])

# Pie de página con nombres
st.sidebar.markdown("""
---
**Equipo del Proyecto:**
- Ana Laura Ruibal Conti
- Mariano Santos
- Nicolás Wicky
- Martín Salamero
""")
# Página de Inicio
if menu == 'Inicio':
    st.title('Análisis de Reseñas Tripadvisor - Parque Xcaret')
    st.markdown("""
    ## Descripción General
    Este proyecto está diseñado para analizar reseñas de productos mediante varias etapas:
                
    - **Web Scraping**: Extracción de datos desde sitios web.
    - **Limpieza de Datos**: Preparación de los datos para su análisis.
    - **Análisis de Sentimientos**: Clasificación de reseñas en positivas, negativas o neutras.
    - **Visualización de Resultados**: Gráficos que muestran las distribuciones y patrones en los datos.
    
    Utiliza el menú de la izquierda para navegar entre las diferentes secciones. Cada sección te guiará a través del proceso completo del análisis de reseñas.
    """)

    # Agregar una imagen descriptiva para la página de inicio
    #st.image("D:/Repos/MCD_Textmining/images/Xcaret-letras.jpg", use_column_width=True)
    st.image("Xcaret-letras.jpg", use_column_width=True)

# Página de Web Scraping
elif menu == 'Web Scraping':
    st.header('Web Scraping')
    st.markdown("""
    Proceso de extracción de reseñas de TripAdvisor.

1.	Definición del alcance y configuración del script: El script tiene como objetivo extraer 1,000 reseñas del parque Xcaret publicadas en TripAdvisor, de un total de 14,500 reseñas disponibles. Para ello, se configura un ciclo que recorre un número determinado de páginas, cada una conteniendo un conjunto de opiniones. El uso de Selenium permite interactuar con el contenido dinámico generado por JavaScript, mientras que ScraperAPI se utiliza para evitar bloqueos durante el scraping.
2.	Generación y manipulación de URLs: La URL base del parque Xcaret en TripAdvisor se utiliza para la primera página de reseñas. Posteriormente, el script genera dinámicamente URLs adicionales para las páginas siguientes, modificando el parámetro “offset” en la URL para acceder a nuevos conjuntos de reseñas, por ejemplo, "Reviews-or10", "Reviews-or20", etc., hasta alcanzar el número deseado de páginas.
3.	Acceso y carga de la página: El script utiliza ScraperAPI para acceder a la página de TripAdvisor evitando bloqueos y restricciones de scraping. Selenium, configurado en modo “headless” para optimizar la extracción, carga la página simulando la interacción de un usuario real. El script espera un tiempo determinado para asegurar que todo el contenido, incluidos los elementos generados dinámicamente, se carguen correctamente.
4.	Extracción de datos con BeautifulSoup: Una vez que la página ha sido completamente cargada, Selenium obtiene el código HTML. BeautifulSoup se utiliza para analizar este HTML y localizar las reseñas y las fechas de publicación, identificando las clases CSS relevantes ("yCeTE" para reseñas y "biGQs" para fechas). Luego, el texto de cada reseña y su fecha correspondiente se extraen y se almacenan en una lista para su posterior procesamiento.

    
    Aquí puedes incluir un código de ejemplo o un resumen del proceso de scraping.
    """)

    st.code("""
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import csv

# Configurar Selenium para usar ScraperAPI
SCRAPER_API_KEY = '8c88a0c1d251d6c0fbfc04f8d15dea51'  # Es una clave para puentear el baneo de la página.

# Configurar opciones de Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ejecutar en modo headless, sin ventana gráfica
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

# Inicializar el WebDriver de Chrome
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

def get_reviews_from_page(url):
    # Construir la URL con ScraperAPI
    scraperapi_url = f'http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={url}'

    try:
        # Navegar a la URL de ScraperAPI
        print(f"Accediendo a la página: {url}")
        driver.get(scraperapi_url)

        # Esperar a que la página se cargue completamente
        time.sleep(10)  # Aumentar el tiempo de espera si el contenido es dinámico

        # Obtener el contenido de la página después de que se haya ejecutado el JavaScript
        page_source = driver.page_source

        # Parsear el contenido con BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Buscar las opiniones usando las clases identificadas
        reviews = soup.find_all('span', class_='yCeTE')  # Clase para los comentarios
        dates = soup.find_all('div', class_='biGQs _P pZUbB xUqsL ncFvv osNWb')  # Clase para las fechas

        # Verificar si se han encontrado opiniones
        if not reviews or not dates:
            print("No se encontraron opiniones o fechas. Verifica las clases o si el contenido es dinámico.")
            return []

        # Extraer y devolver todas las opiniones con sus fechas
        reviews_with_dates = []
        for review, date in zip(reviews, dates):
            reviews_with_dates.append({
                'comment': review.get_text(strip=True),
                'date': date.get_text(strip=True)
            })

        return reviews_with_dates

    except Exception as e:
        print(f"Error durante el acceso a la página: {e}")
        return []

def main():
    # Configurar la cantidad de páginas que deseamos extraer
    total_pages = 600  # Cambia este valor para obtener más o menos páginas
    opinions_per_page = 10  # Opiniones por página

    # Lista para almacenar todas las opiniones
    all_reviews = []

    for page_number in range(total_pages):
        # Calcular el nuevo offset para la página actual
        current_offset = page_number * opinions_per_page

        # Construir la URL de la página actual
        if page_number == 0:
            page_url = 'https://www.tripadvisor.com.mx/Attraction_Review-g150812-d152777-Reviews-Xcaret-Playa_del_Carmen_Yucatan_Peninsula.html'
        else:
            page_url = f'https://www.tripadvisor.com.mx/Attraction_Review-g150812-d152777-Reviews-or{current_offset}-Xcaret-Playa_del_Carmen_Yucatan_Peninsula.html'

        # Obtener opiniones de la página actual
        reviews = get_reviews_from_page(page_url)
        all_reviews.extend(reviews)

        # Mostrar cuántas opiniones hemos recopilado hasta ahora
        print(f"Opiniones recopiladas hasta ahora: {len(all_reviews)}")

    # Imprimir el listado de todas las opiniones recopiladas
    print("\nListado de todas las opiniones recopiladas:")
    for i, review in enumerate(all_reviews, start=1):
        print(f"Review {i}: {review['comment']} - Fecha: {review['date']}\n")

    # Guardar todas las opiniones en un archivo de texto
    with open('todas_las_opiniones.txt', 'w', encoding='utf-8') as file:
        for i, review in enumerate(all_reviews, start=1):
            file.write(f"Review {i}: {review['comment']} - Fecha: {review['date']}\n")

    print(f"Total de opiniones recopiladas: {len(all_reviews)}")

    # Guardar todas las opiniones en un archivo CSV sin comillas dobles
    with open('D:/xcaret1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Review Number', 'Review Text', 'Date'])  # Escribir el encabezado del CSV
        for i, review in enumerate(all_reviews, start=1):
            csvwriter.writerow([i, review['comment'], review['date']])

    print(f"Opiniones guardadas en D:/xcaret.csv")

if __name__ == "__main__":
    main()

    # Cerrar el navegador al final
    driver.quit()

    """, language="python")

# Página de Limpieza de Datos
elif menu == 'Limpieza de Datos':
    st.header('Limpieza de Datos')
    st.markdown("""
    En esta sección, describimos el proceso de limpieza de los datos. La limpieza de datos es un paso esencial antes de realizar cualquier análisis.
    
    Las operaciones incluyen:
    - Eliminar caracteres no deseados.
    - Separar datos en columnas útiles.
    - Transformar datos en formatos apropiados.
    """)

    # Mostrar el DataFrame con los datos limpios
    st.subheader('Datos Limpios')
    st.dataframe(df.head())

# Página de Análisis de Sentimientos
elif menu == 'Análisis de Sentimientos':
    st.header('Análisis de Sentimientos API de OpenAI')
    st.markdown("""
    El análisis de sentimientos es una técnica de procesamiento de lenguaje natural que clasifica las opiniones en **positivas**, **negativas** o **neutras**.
   

 Implementación del análisis de sentimientos:

1.	Envío de las reseñas al asistente de OpenAI: El script utiliza la función submit_message para enviar el texto de cada reseña al asistente de OpenAI, solicitando su análisis de sentimientos. El texto de la reseña se extrae de un archivo CSV y se convierte en un mensaje que se envía a través de la API. Se inicia un hilo para cada mensaje, lo que permite procesar las respuestas de forma estructurada.
2.	Espera y recuperación de resultados: Una vez que el mensaje ha sido enviado, la función wait_on_run verifica el estado del análisis de la reseña, esperando hasta que el asistente haya completado la tarea. Esta función asegura que el script no continúe hasta que el análisis esté finalizado. Después de completarse el análisis, el script recupera los mensajes de respuesta del asistente.
3.	Procesamiento del resultado en JSON: El asistente de OpenAI responde con un formato JSON que contiene la clasificación del sentimiento. El script utiliza la función pretty_print para extraer los datos relevantes de este JSON, verificando que sea un formato válido y asegurando que no haya errores en la respuesta. Los datos extraídos incluyen la "clasificación" del sentimiento y una "valoración" adicional si está disponible.
4.	Almacenamiento de resultados: Una vez que el análisis de sentimientos ha sido procesado y clasificado, el resultado se almacena en el DataFrame original que contiene todas las reseñas. Se agregan dos nuevas columnas al DataFrame: una para la clasificación de sentimiento y otra para la valoración asociada. Esto permite que todas las reseñas, junto con su análisis de sentimientos, se guarden en un archivo CSV para su posterior análisis.
5.	Automatización del proceso: El script automatiza todo el proceso para cada una de las reseñas en el archivo CSV, iterando por todas las filas y aplicando el análisis de sentimientos de manera continua, lo que permite realizar el análisis de grandes volúmenes de datos de forma eficiente.

Clasificación en sentimientos positivos, negativos y neutros:

1.	Extracción de la clasificación del sentimiento: Después de que el asistente de OpenAI analiza una reseña, la respuesta JSON incluye una clasificación que identifica el sentimiento como positivo, negativo o neutro. La función pretty_print se encarga de extraer esta información y validarla.
2.	Posibles clasificaciones: El asistente puede clasificar las reseñas en tres categorías principales:
1.	Sentimientos positivos: Indican satisfacción del visitante con aspectos del parque, como la calidad del servicio, los espectáculos, etc.
2.	Sentimientos negativos: Reflejan insatisfacción del visitante, como críticas a los servicios, las instalaciones o la atención recibida.
3.	Sentimientos neutros: Son aquellos que no muestran una opinión claramente favorable ni desfavorable, indicando una experiencia mixta o neutral.
3.	Asignación de las clasificaciones al DataFrame: Una vez que se extrae la clasificación del sentimiento para cada reseña, se agrega al DataFrame junto con la reseña original. Esta clasificación permite a los usuarios identificar rápidamente la distribución de los sentimientos entre las opiniones.
4.	Uso de la valoración adicional: En algunos casos, la respuesta del asistente también puede contener una "valoración", que es un complemento a la clasificación del sentimiento. Esta valoración podría representar una escala más específica, como una puntuación que cuantifique la intensidad del sentimiento. Si esta valoración está disponible, también se almacena en el DataFrame.
5.	Preparación para análisis visual: Al clasificar las reseñas en sentimientos positivos, negativos y neutros, el script prepara los datos para futuras visualizaciones. Estas clasificaciones permiten generar gráficos, como barras o tortas, que representen la proporción de cada tipo de sentimiento, así como su distribución en función de diversos factores.
            
                
                
                """)

    st.code("""
    import pandas as pd
    import json
    import time
    from openai import OpenAI
    import math

    # Creación del cliente
client = OpenAI(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Función para leer un archivo CSV y devolver un DataFrame
def leer_csv(file_path):
    return pd.read_csv(file_path, delimiter='|')

# Función para guardar un DataFrame en un archivo CSV
def guardar_csv(df, file_path):
    df.to_csv(file_path, index=False)

# Esperar a que el hilo termine
def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.2)
    return run

# Enviar un mensaje al asistente
def submit_message(assistant_id, thread, mensaje):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=mensaje
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

# Obtener la respuesta del asistente
def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

# Crear un hilo y enviar un mensaje
def create_thread_and_run(mensaje, asistente):
    thread = client.beta.threads.create()
    run = submit_message(asistente, thread, mensaje)
    return thread, run

# Función para enviar cada mensaje y almacenar la respuesta
def procesar_filas(df, asistente_id):
    respuestas = []
    for index, row in df.iterrows():
        mensaje = row['Comentario']  
          # Verificar si el mensaje es válido
        if pd.isna(mensaje) or isinstance(mensaje, float) and (math.isnan(mensaje) or math.isinf(mensaje)) or mensaje == "":
            print(f"Mensaje inválido en la fila {index}, se omitirá.")
            clasificacion, valoracion = "No disponible", "No disponible"
        else:
            print(f"Mensaje: {mensaje}")
            thread, run = create_thread_and_run(mensaje, asistente_id)
            run = wait_on_run(run, thread)

            # Verificar si hay un run activo
            if run.status != "completed":
                print(f"El run en el hilo {thread.id} aún no ha terminado.")
            else:
                # Continuar si no hay un run activo
                messages = get_response(thread)
                clasificacion, valoracion = pretty_print(messages)
        
        respuestas.append((clasificacion, valoracion))
    
    df['clasificacion'] = [resp[0] for resp in respuestas]
    df['valoracion'] = [resp[1] for resp in respuestas]
    return df

# Imprimir mensajes, obtiene la clasificación y valoración de la respuesta del asistente y lo devuelve como tupla
def pretty_print(messages):
    
    Prints the messages and extracts classification and rating information from the assistant's response.

    Args:
        messages (list): A list of messages.

    Returns:
        tuple: A tuple containing the classification and rating extracted from the assistant's response.
               Returns ("No disponible", "No disponible") if the assistant's response cannot be decoded.
    
    print("# RESPUESTA API")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
        if m.role == "assistant":
            try:
                # Extraer el contenido JSON eliminando los caracteres de bloque de código y posibles espacios
                raw_content = m.content[0].text.value.strip("```json").strip()
                
                # Verificar si el contenido es un JSON válido
                if raw_content.startswith("{") and raw_content.endswith("}"):
                    response_dict = json.loads(raw_content)
                    clasificacion = response_dict.get("clasificación", "No disponible")
                    valoracion = response_dict.get("valoración", "No disponible")
                    
                    print(f"Clasificación: {clasificacion}")
                    print(f"Valoración: {valoracion}")
                    print("-----------------------------")
                    return clasificacion, valoracion
                else:
                    print("El contenido no es un JSON válido.")
            except json.JSONDecodeError:
                print("No se pudo decodificar la respuesta del asistente.")
    print("-----------------------------")
    return "No disponible", "No disponible"
            
    """, language="python")

# Página de Visualización de Resultados
elif menu == 'Visualización de Resultados':
    st.header('Visualización de Resultados')
    st.markdown("""
    En esta sección puedes visualizar los resultados del análisis de los datos recopilados.
    """)

     # Mostrar los resultados del análisis de sentimientos
    sentimiento_count = df['Sentimiento'].value_counts()
    st.bar_chart(sentimiento_count)



    # Mostrar la distribución de clasificaciones
    st.subheader('Distribución de Clasificaciones')
    clasificaciones = Counter([item for sublist in df['clasificacion'] for item in sublist])
    st.bar_chart(pd.Series(clasificaciones))

    # Mostrar tabla de reseñas
    st.subheader('Tabla de Reseñas')
    st.dataframe(df[['Comentario', 'Fecha', 'Sentimiento']])

    # Normalize ratings and classifications
    df2['valoracion'] = df2['valoracion'].str.lower().str.replace('<', '').str.replace('>', '').str.strip()
    df2['clasificacion'] = df2['clasificacion'].str.replace('<', '').str.replace('>', '').str.strip()

    # Style settings for Markdown
    titulo_principal = "color: white; font-size: 32px; font-family: Arial, sans-serif; text-decoration: underline;"
    subtitulo = "color: white; font-size: 28px; font-family: Arial, sans-serif;"
    etiquetas = "color: white; font-size: 18px; font-family: Arial, sans-serif;"

    # Title and subtitle in Markdown with custom styles
    st.markdown(f"""
        <h1 style='text-align: center; {titulo_principal}'>
        RESEÑA DE REVIEWS TRIPADVISOR
        </h1>
        <h2 style='text-align: center; {subtitulo}'>
        Parque de diversiones XCARET
        </h2>
    """, unsafe_allow_html=True)

    # Configure columns for layout
    #col1, col2 = st.columns(2)

    # Function to calculate ratings percentages
    def calcular_valoraciones(df2):
        valoraciones = df2['valoracion'].str.split(',', expand=True).stack().str.strip().value_counts()
        valoraciones = valoraciones.groupby(valoraciones.index).sum()  # Group by name
        return valoraciones / valoraciones.sum() * 100

    # Calculate overall ratings percentages
    valoraciones_porcentajes = calcular_valoraciones(df2)

    # Pie chart in column 1 for general distribution of ratings
    #with col1:
    st.markdown(f"<h3 style='{etiquetas}'>Distribución general de valoraciones</h3>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots()
    ax1.pie(valoraciones_porcentajes, labels=valoraciones_porcentajes.index, autopct='%1.1f%%', startangle=90,
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    ax1.axis('equal')  # Ensure it's a circle
    st.pyplot(fig1)

    # Dropdown menu and bar chart in column 2
    #with col2:
    st.markdown(f"<h3 style='{etiquetas}'>Distribución de valoraciones por item</h3>", unsafe_allow_html=True)

    # Lista de clasificaciones disponibles para seleccionar
    clasificaciones = ['general', 'naturaleza/atracciones', 'show', 'no clasificable', 'trato personal', 'buffet',
                    'precio', 'animales', 'transporte a hoteles', 'No disponible']

    # Selección de clasificación por parte del usuario
    seleccion_clasificacion = st.selectbox("Seleccione una clasificación:", clasificaciones)

    # Filtrado del dataframe según la clasificación seleccionada
    df_clasificacion = df2[df2['clasificacion'].str.contains(seleccion_clasificacion, na=False)]

    # Calcular las valoraciones de la clasificación seleccionada
    valoraciones_clasificacion = calcular_valoraciones(df_clasificacion)

    # Crear el gráfico
    fig2, ax2 = plt.subplots(figsize=(10, 6))  # Tamaño ajustado del gráfico
    bars = ax2.bar(valoraciones_clasificacion.index, valoraciones_clasificacion, 
                color=['#FF0000', '#FFFF00', '#008000'])  # Colores personalizados

    # Etiquetas y título
    ax2.set_xlabel('Tipo de Valoración', color='black', fontsize=12)  # Etiqueta en el eje X
    ax2.set_ylabel('Porcentaje (%)', color='black', fontsize=12)      # Etiqueta en el eje Y
    ax2.set_title('Distribución de Valoraciones por Clasificación', color='black', fontsize=14)  # Título del gráfico
    ax2.tick_params(axis='x', colors='black', rotation=45)  # Rotación de etiquetas en el eje X
    ax2.tick_params(axis='y', colors='black')               # Colores de las etiquetas en el eje Y

    # Agregar etiquetas de cantidad en cada barra
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}%', ha='center', va='bottom', color='black')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig2)
