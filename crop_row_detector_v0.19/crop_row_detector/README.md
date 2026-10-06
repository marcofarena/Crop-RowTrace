# Detección de Hileras de Cultivo — plugin de QGIS

Algoritmo de **Processing** para QGIS que detecta hileras de cultivo en un
ortomosaico RGB (drone o satélite) sin necesidad de entrenar ningún modelo.

## Cómo funciona

Pasos comunes a los dos métodos:

1. Lee las bandas R, G, B y la **máscara de píxeles válidos** de GDAL
   (banda alfa o valor nodata). Los píxeles transparentes de un raster
   recortado se ignoran en todo el cálculo.
2. Calcula el índice de vegetación **ExG cromático = (2G − R − B) / (R + G
   + B)**, la definición original de Woebbecke et al. (1995) sobre
   coordenadas cromáticas. Mide *qué tan verde* es un píxel,
   independientemente de cuánto brilla. Esto importa: en imágenes
   satelitales o aéreas con canopia densa, la canopia (y su sombra) es
   **más oscura** que la entrehilera, y el ExG sin normalizar (2G − R − B,
   usado hasta la v0.6) escala con el brillo, así que un suelo claro con
   algo de pasto da más que una canopia verde oscura y las "hileras" caen
   sobre la entrehilera. Las bandas se suavizan levemente (σ = 1 px) antes
   del cociente, porque en píxeles casi negros la cromaticidad está
   cuantizada y es ruido. La versión sin normalizar queda disponible en
   "Parámetros avanzados".
3. Binariza la imagen (umbral automático de Otsu, 1979, calculado solo
   sobre los píxeles válidos y entre los percentiles 0,5 y 99,5 del índice;
   o umbral manual). La máscara binaria se usa para el % de cobertura y el
   conteo de plantas, y como entrada del método Hough.

### Método 1 — Perfil de proyección orientado por FFT (default)

Pensado para hileras **rectas y paralelas** (viñedo, frutal, cultivos en
línea), que es la geometría típica de un lote implantado.

4. **Orientación y separación por FFT.** Un patrón de franjas paralelas y
   equiespaciadas concentra la energía de su espectro de Fourier 2D en un
   pico ubicado a la frecuencia 1/T (T = separación entre hileras), en la
   dirección *perpendicular* a las franjas. La posición de ese pico da a la
   vez el rumbo de las hileras y su separación. Se promedian los espectros
   de varias ventanas repartidas por el lote (método de Welch, 1967) con
   ventana de Hann. El ángulo se afina luego buscando el que maximiza el
   contraste del perfil perpendicular (con el ángulo exacto, cada franja
   del perfil cae entera sobre hilera o entera sobre entrehilera).

   **Espectro blanqueado (hileras finas).** Una imagen de campo tiene un
   fondo de potencia que cae como 1/f: manchones de vigor, de suelo o de
   riego de decenas de metros. Con hileras finas (hortalizas a menos de
   1 m) ese fondo supera en potencia al pico de las hileras y el máximo del
   espectro cae en un manchón (visto: 36 m en vez de 0,85 m). Un patrón
   periódico sobresale de *su anillo* de frecuencia (la mediana de la
   potencia a esa misma frecuencia en todas las direcciones): en los 23
   lotes de prueba con hileras bien detectadas, entre 115 y 75.000 veces;
   el manchón, 2,9 veces. Si el máximo no sobresale al menos 20 veces de su
   anillo, se busca el pico en el espectro dividido por la mediana de cada
   anillo. Ahí solo cuentan los picos con **potencia propia** (al menos
   0,2% de la del máximo crudo) y de al menos 8 ciclos por ventana: el
   granulado del remuestreo de la imagen (2-4 px, casi paralelo a los ejes
   del raster) sobresale miles de veces de su anillo con una potencia
   ínfima (0,003-0,02% del máximo; las hileras, 0,7-2%), y a baja
   frecuencia el anillo tiene tan pocos bins que un manchón puede
   sobresalir 20 veces. Se toma el pico más fuerte y, entre los picos
   fuertes **en su misma dirección** a una frecuencia n veces menor, el de
   menor frecuencia: la **fundamental** de las hileras, no sus armónicos
   (T/2, T/3...), que con un surco angosto entre camellones pueden ser más
   altos (en un lote hortícola de camellones cada 5,1 m el más fuerte era
   el cuarto armónico, 1,28 m). Con hileras finas el ángulo se afina además sin la media móvil
   de 2T del perfil, que contiene los manchones y no el patrón de hileras
   (sin eso, 8,1° en vez de 7,1°).

   **Marco de plantación (árboles sueltos en grilla).** En un frutal en
   marco rectangular (o tresbolillo) los árboles quedan alineados en *dos*
   direcciones: las hileras (cada T) y, cruzando, los árboles de hileras
   vecinas (cada distancia entre plantas). El espectro tiene dos picos, y
   el más fuerte puede ser el cruzado (visto: frutal de 5 × 4 m con pasto,
   en el verdor 4,02 m con fuerza 1,00 contra 4,97 m con 0,70): las líneas
   salían perpendiculares a las hileras, uniendo un árbol de cada hilera.
   Si hay un segundo pico con al menos 30% de la fuerza del máximo, a 45° o
   más de él y con un período entre 1,1 y 3 veces mayor, se toma ese como
   las hileras: la distancia entre hileras es mayor que la distancia entre
   plantas (por ahí pasa el tractor). El log lo avisa con las dos
   alineaciones. Si se indica la **Distancia entre hileras**, entre los
   picos de la ventana de búsqueda se toma el más fuerte a ±10% de ese
   valor (en un marco de 5 × 4 m los 4 m quedan a −20%), y si no hay
   ninguno, el más cercano. Solo picos verdaderos del espectro: recortado
   a la ventana, el flanco de un pico fuerte de afuera quedaba como
   "máximo" en su borde (visto en un olivar de 12,25 m: con 12 m
   indicados daba 7,8 m).

   **Marco entre índices.** El verdor y el brillo pueden ver alineaciones
   distintas. En un olivar de 12,25 × 7,8 m el verdor mostraba sobre todo
   la alineación cruzada de los árboles (7,4 m; las hileras, apenas 12% de
   esa fuerza) y el brillo las hileras (árboles con su sombra contra el
   suelo claro de la entrehilera): las líneas salían giradas 90°. Si el
   pico del brillo está a 45° o más del elegido en el verdor y su período
   es entre 1,1 y 3 veces mayor (y sobresale de su anillo), se toma el del
   brillo, con el mismo criterio: la de mayor separación son las hileras.
   También si el verdor ya había elegido entre dos de sus alineaciones.
   Con eso el olivar da 36,2°, 12,25 m y 22 de 23 hileras.
5. **Centros de hilera.** El ExG se remuestrea a una grilla rotada en la
   que las hileras quedan verticales; el promedio por columna (perfil
   perpendicular) tiene un máximo en el centro de cada hilera.

   **Franja de la hilera: copa cerrada.** El verdor cromático ubica la
   hilera en la franja *más verde*. En viñedos y frutales con suelo entre
   hileras esa franja es la canopia (oscura, con su sombra). En un frutal
   de **copa cerrada** no hay suelo entre hileras: las copas iluminadas
   ocupan casi todo el ancho y el hueco entre ellas queda en sombra. Esa
   sombra es verde oscura y tiene el verdor cromático *más alto* (en un
   píxel oscuro, la luz que queda es la que filtran y reflejan las hojas),
   así que la línea caía entre copas. Se reconoce porque la franja a media
   separación también es vegetación, es más clara y es más verde en valor
   absoluto (2G − R − B): son las copas al sol. Criterio: separación de
   frutal (≥ 3,5 m), verdor de la franja a media separación ≥ 30% del de la
   más verde (con suelo desnudo entre hileras da 9-20%; con copa cerrada,
   59%) y verde absoluto ≥ 1,1 veces (0,1-0,4 con suelo; 1,4 con copa
   cerrada). En ese caso las hileras se ubican con el verde absoluto; el
   brillo no se usa como evidencia auxiliar (marca sombras, y el suelo de
   una calle, más claro que las copas, pasaba el control de nivel); el
   nivel mínimo de un tramo baja a 0,25 (la sombra entre copas es la
   referencia de entrehilera y está por encima del suelo); y el centro de
   cada hilera se reubica con el perfil suavizado a T/4, porque una copa
   ancha y pareja no tiene un máximo definido. Solo con separación de
   frutal: en un viñedo con cubierta verde entre hileras la vid es la
   franja oscura y el pasto la clara, con la misma firma. El parámetro
   **Franja de la hilera** permite forzar uno u otro criterio.

   **Pasto entre árboles sueltos.** Un frutal joven con pasto verde entre
   hileras tiene la misma firma que la copa cerrada (visto: verdor a media
   separación 76%, verde absoluto 1,4 veces) y las líneas iban sobre el
   pasto. Cuando el lote muestra un marco de plantación (ver el paso 4), se
   sabe dónde están los árboles: en la franja que "late" al ritmo de las
   plantas a lo largo de la hilera (cada árbol, un máximo; entre árboles,
   pasto). El pasto entre hileras es parejo a lo largo de la hilera. Si a
   media separación late menos de la mitad que en la franja más verde, los
   árboles son la franja más verde y no se aplica copa cerrada (en el
   frutal visto, 18%; el log lo explica). En un frutal de copa cerrada en
   seto no se ve marco y el criterio no cambia.
6. **Tramos presentes.** A lo largo de cada hilera se compara el ExG
   medio de una banda centrada en la hilera (de ancho T/2 + 1 píxel) con
   el de las dos entrehileras vecinas (a ±T/2). El ancho de la banda
   varía de forma continua con T (las columnas del borde pesan según la
   fracción que cae dentro) y las entrehileras se interpolan: así el
   resultado no salta por un error de décimas de píxel en la separación
   estimada.
   Es un contraste **local y relativo**: no depende del vigor absoluto de
   la zona, así que no pierde las zonas de menor vigor como un umbral
   global. La hilera se considera presente donde ese contraste supera
   `Sensibilidad` × ruido **y** el centro es más verde que *ambas*
   entrehileras (un borde de árbol o de camino da contraste de un solo
   lado). El ruido se estima con la cola negativa de la distribución del
   contraste, que solo puede venir de ruido: una hilera real nunca es
   menos verde que su entrehilera.
7. **Continuidad del patrón.** Donde el verdor se debilita (vides de
   menor vigor, pasto verde en la entrehilera que invierte el contraste de
   verdor, plantas de borde) la hilera se sigue mientras continúe su patrón
   de **brillo**: la franja de la hilera más oscura (o más clara) que
   *ambas* entrehileras, con el mismo signo que esa hilera tiene donde el
   verdor es claro. Una mancha uniforme, como la copa y la sombra de un
   árbol, no es más oscura que ambos vecinos a ±T/2 y no cuenta. Con el
   mismo criterio se agregan hileras sin pico propio de verdor
   (típicamente las de borde del lote), extendiendo la grilla periódica
   mientras el patrón de brillo continúe. El brillo cuenta como evidencia
   donde su contraste supera `Sensibilidad` × 10% del contraste típico
   (la mediana) de las hileras bien vistas por el verdor, o sea el 30% con
   el valor por defecto. No se usa la cola negativa, como en el verdor:
   en esas hileras el contraste de brillo casi nunca es negativo, y los
   pocos valores negativos son anomalías (un hueco, un árbol), no ruido.

   **Entrehileras alternadas** (rastra o cobertura hilera por medio): de
   un lado de cada hilera el suelo rastreado es claro y del otro la
   cobertura se ve casi igual a la vid, así que la exigencia de ser más
   oscura que *ambas* entrehileras cortaba las hileras (visto: en un
   viñedo de 1,3 ha, 0,8 km de puntas cortas; la vid 8-20 ruidos más
   oscura que el lado rastreado y 0 del lado cubierto). Se reconoce en el
   brillo porque el lado claro cambia de costado de una hilera a la
   vecina: en al menos la mitad de las hileras hay pares vecinos con una
   diferencia entre sus dos entrehileras de 2 ruidos o más, y en al menos
   el 90% de ellos el signo se invierte. Ahí se acepta que la hilera no se
   distinga del lado cubierto (tolerancia de 3 ruidos). Viñedo: F1 0,942 →
   0,968. Pasa en 4 de los 43 lotes de prueba.

   El brillo cumple además dos funciones de control:

   - **Índice principal cuando el verdor no sirve.** Si en un lote la
     periodicidad del brillo es más de 3 veces la del verdor (pico
     espectral) y ambas coinciden en rumbo y separación, las hileras se
     ubican con el brillo, como las **franjas oscuras**. Pasa cuando la
     canopia casi no es verde en la imagen (vides sin hojas, canopia en
     sombra profunda, imagen de otra fecha): el verdor de píxeles casi
     negros es ruido de cuantización (p. ej. RGB 0,1,0 da ExG = 2) y los
     centros de hilera caían en cualquier lado. Se toma la franja oscura
     porque la canopia más su sombra reflejan menos que el suelo; en la
     finca de prueba el color medio lo confirmó en todos los lotes con
     verdor medible. El log avisa cuando pasa.

     Con el ángulo ya afinado se revisa esa elección con el perfil a lo
     ancho de las hileras (verdor de la franja más verde y dónde cae la
     franja más oscura respecto de ella):
     - **Canopia no verde** (verdor de la franja más verde < 0: sin hojas,
       imagen de otoño, canopia rojiza): la franja "más verde" es solo la
       menos rojiza, a menudo el suelo, y las líneas caían sobre la
       entrehilera y sobre calles y callejones. Se usa la franja oscura
       aunque el brillo no sea 3 veces más fuerte. Visto en un lote de
       franjas oscuras rojizas: verdor −0,08; en viñedos sin hoja, −0,08 a
       −0,004.
     - **La franja oscura es sombra**: si el brillo es más fuerte pero la
       canopia es claramente verde (verdor ≥ 0,10) y la franja oscura está
       corrida más de 0,15 T de la más verde, la oscura es la sombra al
       costado de la canopia y se usa la más verde. Visto en un frutal en
       seto (verdor 0,27, corrimiento 0,17 T): de 27 a 57 de 61 hileras,
       en el centro de la banda de canopia. En viñedos las dos franjas
       coinciden (0 a 0,08 T) y no cambia nada.
     - Entre 0 y 0,10 (canopia apenas verde: frutales jóvenes o ralos, el
       suelo entre árboles diluye el promedio): si la franja oscura está
       corrida más de 0,15 T de la más verde, la más verde es el suelo o la
       cobertura entre hileras y se usa la **oscura**. Visto en un frutal
       joven con cobertura clara entre hileras (verdor 0,02, corrimiento
       0,25 T): las líneas iban en el medio de la entrehilera; otros dos
       frutales jóvenes con la misma firma ya iban a la oscura porque su
       brillo era 3 veces más fuerte. Si las dos franjas coinciden se sigue
       con el índice elegido (cambiarlo acortaba 10% las líneas en dos
       frutales de la finca). Con las franjas separadas se avisa para
       revisar (ver "Aviso de revisión").

     Con la franja oscura como índice, el verdor queda como imagen
     auxiliar y, además de extender tramos, también los **siembra**: en el
     lote de canopia rojiza la franja oscura se veía débil en muchas
     hileras mientras el verdor (la franja "menos rojiza" entre plantas)
     las veía todas, a 10-17 veces el ruido; 36 hileras quedaban sin línea.
     La posición sigue siendo la de la franja oscura. Hileras sin línea en
     ese lote: de 40 (5,0 km) a 8 (0,5 km), sin tramos falsos nuevos.
   - **Verificación de posición.** Con el verdor como índice, en una
     franja del lote donde la entrehilera es más verde que la hilera
     (pasto o maleza entre hileras, vides debilitadas) el máximo de verdor
     cae sobre la *entrehilera*. El patrón de brillo no se invierte: si en
     la mayoría de las hileras la hilera es la franja oscura, un centro
     que cayó sobre una franja clara se mueve a la franja oscura más
     cercana, y el brillo también siembra tramos en hileras bien ubicadas
     cuyo verdor no alcanza. El log informa cuántas hileras se
     reubicaron. Si la franja oscura se juzgó **sombra** al costado de la
     canopia (ver arriba), el centro no va a la franja oscura sino a la
     canopia: a la franja oscura menos el corrimiento de la sombra, medido
     en todo el lote. Visto en el frutal en seto: 19 líneas quedaban a 1 m
     del centro de la canopia, del lado de la sombra (y sin reubicar se
     perdían 15 hileras, con el centro de verdor en la maleza).
8. **Extremos.** El umbral es por histéresis (como en Canny): un tramo
   necesita una racha de al menos T por encima de `Sensibilidad` × ruido,
   y se extiende mientras la evidencia siga por encima de 1 × ruido. Cada
   extremo se ubica a **media altura** de la amplitud local de la hilera:
   el suavizado a lo largo de la hilera convierte su final abrupto en una
   rampa, y con un filtro de caja el valor suavizado vale exactamente la
   mitad del escalón en la posición real del borde. Los fragmentos
   terminales más cortos que la longitud mínima (p. ej. un árbol verde en
   la prolongación de la hilera, pasando la cabecera) no se unen al resto,
   **salvo** que no pasen del borde de lo plantado que forman las hileras
   vecinas (+T/2): entonces son la punta de la propia hilera, separada por
   una o dos plantas débiles o faltantes (visto: un corte de 0,4 m y se
   perdían los últimos 10 m). Sin un borde consistente de las vecinas no se
   suman.
   **Microcortes.** Un corte de la evidencia débil de hasta T/4 (una
   planta chica, un píxel de ruido) no parte el tramo. Partido, lo que
   seguía tenía que sembrarse solo con una racha fuerte de largo T, y en un
   frutal joven (árboles chicos, contraste intermitente) se perdía entero:
   visto, un corte de 0,4 m y 60 m de hilera clara sin línea.
   **Árboles sueltos muy separados.** En un olivar a 12 m entre hileras
   con árboles cada 7,8 m, cada árbol da una racha fuerte de 4-7 m y nunca
   una de 12: hileras enteras quedaban sin sembrar (visto: una de 256 m) y
   otras se cortaban tras un árbol faltante. Por eso (1) un tramo también
   se siembra si sus rachas fuertes *suman* T, (2) una racha débil con algo
   de evidencia fuerte, a menos del hueco máximo de un tramo sembrado, se
   le une (la continuación tras uno o dos árboles faltantes), y (3) el hueco
   máximo (un parámetro en píxeles: 60 px son 7,5 m a 0,125 m/px) es al
   menos 1,25 separaciones entre hileras. En ese olivar, de 3,3 a 4,0 km de
   línea y ninguna hilera sin marcar; en viñedos y en los frutales de 6 m
   de la finca de prueba casi no cambia nada.

   **Planta por planta.** La evidencia promediada a lo largo de T no ve
   las plantas chicas (replantes, plantas de menor vigor, típicamente en
   la punta de la hilera): una vez promediadas con los huecos entre ellas
   no alcanzan la evidencia de las plantas grandes, y el extremo a media
   altura las dejaba afuera. Por eso cada hilera también se mira a escala
   de planta (T/3), con el mismo criterio (el centro más verde que *ambas*
   entrehileras, por encima de `Sensibilidad` × su ruido). Las plantas a
   menos de T una de otra forman una secuencia; una secuencia que toca un
   tramo (o queda a menos de T de él) lo prolonga o une dos tramos, y una
   secuencia suelta cuenta sola si mide al menos 3T (una hilera entera de
   plantas chicas). Un árbol aislado en la cabecera, a más de T de la
   última planta, no se une.
   **Nivel.** Además del contraste con sus costados, cada tramo tiene que
   parecerse a una hilera en su *nivel* absoluto: en el índice o en el
   brillo, tiene que estar al menos a mitad de camino entre el nivel típico
   de la entrehilera y el de la hilera del lote (medidos en las hileras que
   el índice ve con claridad). Una huella de tractor, una sombra débil o un
   borde de calle son algo más oscuros (o verdes) que la calle a sus lados,
   pero quedan muy lejos del nivel de una hilera (en la finca de prueba,
   entre −28% y 46% del camino; las hileras, 86-110%). Importa cuando el
   contorno del lote incluye parte de la calle. En tramos de más de 3T el
   nivel se mide sin la media separación de cada punta (la rampa de la
   punta es más débil y bajaba el promedio justo por debajo del umbral).
   Si el tramo entero no
   llega, se prueban por separado sus partes con evidencia continua y se
   conservan las que sí: una parte más pobre de la hilera bajaba el
   promedio y se perdía la hilera entera. En las posiciones **agregadas**
   (grilla del marco o continuidad del patrón, las de borde), un tramo con
   más de 4 veces el contraste típico de una hilera no es la hilera de
   borde sino una **cortina** o una fila de árboles junto al lote, y se
   descarta (visto: una cortina junto a un viñedo, 10,7; las posiciones
   agregadas de 31 lotes, hasta 2,1). Lo mismo para la **hilera extrema**
   de cada lado aunque tenga pico propio: se descarta si su nivel supera
   3 y 2,5 veces el percentil 99 de las hileras interiores del lote (así,
   relativo: en un viñedo de canopia casi negra las interiores llegan a
   5,8). Visto: árboles de la calle junto a un frutal, 31,8 (interiores
   hasta 3,5), y la fila de árboles del borde de un viñedo, 3,8
   (interiores hasta 1,2); las hileras extremas reales de 31 lotes, hasta
   2,1.
   **Plantas chicas o ralas** (hileras de borde de un frutal, replantes):
   su tramo tiene contraste claro con las dos entrehileras, pero su nivel
   promedio queda cerca del de la entrehilera, porque entre planta y
   planta hay suelo. Un tramo descartado por nivel se recupera si (1) en
   el **brillo** es algo más oscuro que la entrehilera típica (al menos
   20% del camino: la copa y la sombra de un árbol, aunque sea chico; en
   la finca de prueba, +27% a +47%), (2) en el otro índice no queda más
   allá de la entrehilera del lado del suelo desnudo en más de 50% (una
   huella o un borde sobre la calle, suelo desnudo y claro, queda en
   −100% a −240%) y (3) está dentro del largo que ocupan las hileras
   vecinas (±T). Unos arbustos claros sueltos sobre un sendero, al borde
   del lote, quedan en −11% a +12% en el brillo y no se recuperan. Sin
   imagen auxiliar, o con hileras finas, no se recupera nada.
8b. **Grilla del marco de plantación.** El marco es regular, así que cada
   posición a un múltiplo de T de las hileras ubicadas (en los huecos entre
   ellas y hacia afuera, hasta el borde del lote) se prueba como hilera
   aunque el perfil perpendicular no muestre ahí un máximo: una hilera de
   plantas chicas o ralas (típicamente la de borde) casi no pesa en el
   perfil promedio del lote. La prueba es la misma de siempre, a lo largo de
   la hilera: en una calle o en suelo desnudo no aparece ningún tramo. La
   posición que se prueba es la del marco exacto; la pasada 2 la reubica
   con la evidencia de la propia hilera (antes se "afinaba" al máximo del
   perfil de todo el lote, donde una hilera que ocupa solo parte del largo
   no pesa, y quedaba a 0,6-0,8 T o a 1,2-1,6 T de su vecina).
8c. **Hileras finas (hortalizas).** Con menos de 6 px por hilera, el
   suavizado pensado para hileras de ~10 px las borra: el verdor se calcula
   sobre bandas suavizadas σ = 1 px, que a un período de 4,3 px deja pasar
   el 34% del patrón, y la banda central (T/2 + 1 px) cubre el 73% del
   período y deja pasar otro tercio. En esos lotes el verdor se recalcula
   con σ = 0,5 px, y el medio píxel extra de la banda central se reduce
   hasta 0 en T = 4 px (sin cambios para T ≥ 8 px).
   **Modo patrón.** Si aun así menos del 30% de las hileras ubicadas tiene
   evidencia propia (su contraste apenas supera el ruido: en una hortaliza
   a 0,85 m sobre una imagen de 0,2 m, la relación señal/ruido de una
   hilera es ~1,2 por metro y hacen falta ~40 m para llegar a 3), y el lote
   tiene al menos 10 hileras, cada hilera se completa donde el patrón
   conjunto de 5 hileras vecinas a lo largo de 10 separaciones es
   significativo (los tramos propios conservan sus extremos): una
   prueba t con la varianza de la propia ventana (la textura de árboles o
   techos es mucho mayor que la del cultivo), sobre el contraste contra
   *ambos* costados, en el verdor o en el brillo (en una parte del lote las
   hileras se ven por uno y en otra por el otro). La posición de cada línea
   es la del perfil del lote, los extremos tienen la precisión de la
   ventana (~10 separaciones) y no se ven fallas de plantas. En viñedos y
   frutales no se activa: ahí el 90-100% de las hileras tiene evidencia
   propia.
9. El centro de cada hilera se reubica por bloques a lo largo de ella
   (solo donde está presente), con una pendiente propia si es
   estadísticamente significativa. Se unen huecos cortos (plantas
   faltantes) y se descartan tramos cortos.

   **Seguimiento lateral (hileras levemente torcidas).** Una recta no
   sigue una hilera plantada a mano que se curva, ni la distorsión local
   del ortomosaico. Donde la hilera real se aparta ~T/4 de su recta, la
   banda central cae sobre el borde de la entrehilera y los costados casi
   sobre las hileras vecinas: el contraste se invierte y ese tramo se
   perdía (visto: 0,6 m de apartamiento a lo largo de 45 m en una vid a
   2 m). Por eso cada hilera que ya se ve sobre su recta (al menos la
   longitud mínima) se mide también corrida hasta 0,3 T a cada lado, pero
   nunca más de 0,6 m (lo que se aparta una hilera plantada es una
   cantidad en metros: en una vid a 2 m son 0,3 T; en un frutal a 6 m,
   0,3 T serían 1,8 m y el camino se enganchaba en arbustos al costado de
   la hilera). Por bloques de largo T se elige el camino con más evidencia
   de los dos lados que sea continuo (programación dinámica: a lo sumo un
   nivel de corrimiento por bloque, con un costo por cambio de nivel y
   otro por apartarse de la recta, así que una hilera recta, o una zona
   sin hilera, se queda en la recta). La evidencia se mide sobre ese
   camino. La línea de salida sigue siendo una recta por tramo; si el
   camino se aparta más de 0,2 T, la recta del tramo se ajusta a él. No se
   aplica a las hileras agregadas por la grilla o por continuidad del
   patrón (bordes del lote, junto a calles y cortinas: ahí siempre aparece
   alguna sombra de costado), ni a hileras de menos de 6 px de separación
   (el camino seguiría al ruido). El log informa cuántas hileras se
   siguieron y cuántos tramos se reajustaron.

   También completa hileras que cruza en diagonal una franja oscura
   (maleza, una acequia): donde la franja cae sobre la entrehilera, la
   hilera vecina perdía el contraste de ese lado y quedaba un hueco.
10. **Ajuste al contorno de la parcela** (ver abajo).

La idea de localizar hileras por máximos del perfil de vegetación sin
segmentar la imagen es la de Søgaard y Olsen (2003); acá se combina con la
estimación espectral de la orientación para no depender de bordes.

### Método 2 — Canny + Hough (método anterior)

4. (Opcional) Aplica un **cierre morfológico** antes de Canny — ver sección
   de viñedo/frutal abajo.
5. Detecta bordes con **Canny** y ajusta líneas rectas con la
   **Transformada de Hough probabilística** (`cv2.HoughLinesP`, Matas et
   al., 2000).
6. (Opcional, activado por default) **Filtra los segmentos por dirección
   dominante**: descarta los que no sigan la orientación predominante del
   conjunto.
7. **Fusiona los segmentos de una misma hilera** en una sola línea: agrupa
   los tramos con ángulo similar, cercanos perpendicularmente (mismo
   "carril") y cercanos a lo largo de la hilera (sin saltar huecos grandes
   sin relación), y ajusta una línea única por grupo con PCA, extendida al
   alcance real de todos sus segmentos.

**Por qué dejó de ser el default:** con hileras angostas y muy juntas (p.
ej. viñedo con 2 m entre hileras a ~0,2 m/px: hileras de 4-5 px de ancho,
período de ~9 px), Canny genera dos bordes por hilera separados por pocos
píxeles, y Hough con un `max_line_gap` mayor que el período "salta" de una
hilera a la siguiente: cualquier recta que *cruce* las hileras encuentra
un píxel de borde cada pocos píxeles y la acepta como línea continua. Esas
rectas transversales además son más largas que las hileras (atraviesan
toda la parcela), así que dominan la dirección dominante ponderada por
longitud: el detector termina eligiendo la dirección **perpendicular** a
las hileras. En un raster real de viñedo (0,22 m/px) la dirección elegida
fue 21,6° contra 112° reales, y con el ángulo correcto impuesto a mano
Hough solo recuperó 33 de ~175 hileras. El método de perfil, sobre el
mismo raster, ubica las 172 hileras con el rumbo correcto, una línea
continua por hilera.

### Varios lotes en el mismo raster

Si la capa de contorno tiene **varios polígonos** (uno por lote), con el
método de perfil **cada lote se procesa por separado**: usa solo los
píxeles de adentro de su polígono, tiene su propia orientación, separación
y alineación de hileras, y sus líneas se recortan a él. La salida suma el
campo `lote` (1..N en el orden de la capa) y `hilera` se numera dentro de
cada lote.

Esto resuelve tres problemas de las escenas con varios lotes:

- **Cultivos u orientaciones distintas** (p. ej. un frutal y un viñedo, o
  lotes con trazados distintos): sin lotes, la FFT elige *una*
  orientación para todo el raster y el resto no se detecta.
- **Lotes uno detrás del otro en la dirección de las hileras**: las
  hileras de un lote no tienen por qué estar alineadas con las del
  siguiente. Sin lotes, la posición de las hileras se calcula una sola vez
  para todo el raster y solo puede coincidir con uno de los dos.
- **Calles entre lotes**: quedan fuera de todos los polígonos, así que
  ninguna línea las cruza.

Desde la v0.13 también un **único polígono** se procesa así: la
orientación y la separación se estiman solo con sus píxeles. Antes, con un
solo polígono, se estimaban con todo el raster, y en un recorte con un
frutal y una hortaliza el frutal fijaba el rumbo de la hortaliza (8,1° en
vez de 7,1°).

**Distancia entre hileras por lote.** La 'Distancia aproximada entre
hileras' vale para todos los lotes; con cultivos de separación muy
distinta (un frutal a 4,5 m y una hortaliza a 0,85 m) no sirve para los
dos. El parámetro **Campo de distancia entre hileras por lote** toma el
valor (en metros) de un campo de la capa de lotes: donde es > 0 reemplaza
a la distancia general; vacío o ≤ 0, se usa la general (o el automático).

Un objeto multiparte cuenta como un solo lote. Los polígonos pueden estar
en cualquier CRS (se reproyectan solos). Si un polígono no toca el raster,
se omite con un aviso, así que una misma capa de lotes sirve para varios
rasters de la misma finca.

### Ajuste al contorno de la parcela

Parámetro **Contorno de la parcela o lotes** (capa de polígonos, opcional;
se reproyecta sola al CRS del raster). Si no se da, se usa el área con
datos del raster (máscara alfa/nodata vectorizada), que en un raster
recortado por la parcela es el mismo contorno.

- **Recorte estricto a lo largo de las hileras** (ambos métodos): toda
  línea se intersecta con el contorno; en las cabeceras nada queda afuera.
- **Hileras de borde** (método de perfil): es común que el borde del lote
  se dibuje *sobre* la primera y la última hilera (clickeando sus
  plantas). Esa hilera queda cortada por el contorno, y además su
  entrehilera exterior queda afuera, sin la cual no se puede medir su
  contraste: hasta la v0.11 se perdía entera. Ahora cada lote se muestrea
  hasta una separación T afuera de su polígono, **solo a lo ancho de las
  hileras**, y una hilera cuyo eje queda sobre el borde o hasta 0,4 T
  afuera se considera la hilera de borde del lote y se conserva entera. Se
  recorta con la paralela a 0,4 T hacia adentro del lote, así que las
  cabeceras siguen siendo estrictas. Una hilera es de borde si el contorno
  le deja menos de la mitad del largo que le deja a esa paralela, o si la
  corta en varios pedazos (un borde dibujado sobre ella, que la cruza de un
  lado al otro). La hilera siguiente hacia afuera estaría a más de 0,5 T y
  no entra: con dos lotes contiguos sin calle, separados por un borde en
  la mitad de la entrehilera, ninguna hilera se duplica. Además, una
  hilera de borde tiene que estar **en el marco de plantación**: a un
  múltiplo de la separación (±T/4) de la hilera interior más cercana. Si no,
  es otra cosa a lo largo del borde (un cerco, una cortina, una fila de
  árboles de la calle) y se descarta. Visto en un frutal: un cerco a 2,4 T
  de la última hilera, con su eje 1,5 m afuera del polígono.
- **Extremos en el borde de lo plantado** (método de perfil): cada hilera
  termina donde termina lo plantado, sin entrar en la calle de cabecera.
  Como el borde plantado es continuo, el extremo de cada hilera debería
  estar alineado con los de sus vecinas: se ajusta una recta local con los
  extremos de hasta 4 hileras a cada lado y, si las vecinas forman un
  borde consistente y el extremo se aparta más de media separación entre
  hileras y más de 3 veces el desvío propio de ese borde (y no más de 3
  separaciones), se lo lleva a ese borde. Las puntas de las vecinas varían
  (plantas de la punta más o menos grandes): hasta la v0.16 se exigía un
  borde casi perfecto (desvío medio ≤ T/4) y con puntas que variaban
  ±0,5 m no se alineaba nada (visto: una hilera 3,5 m metida en la calle de
  cabecera de un viñedo). En la finca de 21 lotes, las puntas metidas en
  la calle bajan de 5 (28 m) a 3 (19 m) según la auditoría. Así se
  recortan hileras que se meten en la calle y se completan las que quedan
  cortas por una o dos plantas débiles en la punta. Donde el borde tiene
  un escalón real (un rincón sin plantar) las vecinas no forman un borde
  común y el extremo no se toca.
- **Extensión hasta el contorno** (método de perfil, **default 0 = no
  extender**): prolonga cada hilera, sobre su propia recta, hasta el borde
  del contorno si le faltan como máximo **Extender hasta el contorno**
  metros. Usalo solo si el contorno que das es el del **área plantada**:
  con el contorno del lote (que incluye las calles de cabecera) las líneas
  atravesarían las calles y quedarían más largas que las hileras reales.

### Salida

Las líneas se convierten a coordenadas del mapa con el geotransform del
raster y se exportan como capa vectorial con longitud y ángulo de cada
línea (y el número de `hilera` en el método de perfil).

### Aviso de revisión (campo `revisar`)

El complemento toma solo varias decisiones que pueden salir mal sin que
el resultado se vea raro: el rumbo, qué franja es la hilera, si hay copa
cerrada. En la evaluación ciega de la v0.16, con los valores por defecto,
3 de 6 lotes salieron mal y el log no marcó ninguno como dudoso. Desde la
v0.17, cuando una de esas decisiones queda con poco margen, el log lo
avisa en rojo al final del lote ("REVISAR este lote…"), resume al final
qué lotes revisar, y cada línea lleva el motivo en el campo `revisar`
(vacío = sin dudas):

- `rumbo`: marco casi cuadrado (la separación entre hileras es menos de
  1,3 veces la de plantas: la elección de la dirección de mayor
  separación tiene poco margen), o pico espectral bajo.
- `franja`: la franja más verde y la más oscura están en lugares
  distintos y la elección quedó en la zona gris (canopia apenas verde) o
  a un paso del umbral de corrimiento; o se aplicó copa cerrada
  automáticamente.
- `cobertura`: menos del 70% de las hileras ubicadas tiene línea (en el
  frutal en seto con las líneas sobre la sombra, 27 de 61).
- `cuadros`: en parte del lote las hileras van a otra separación. El
  complemento usa una sola separación y un solo rumbo por lote, así que un
  polígono que junta cuadros de plantación distintos sale bien en el cuadro
  dominante y mal en los otros (líneas intercaladas, hileras sin marcar).
  Se mide en ventanas de ~8 x 8 separaciones: en cada una, la separación
  con el patrón de franjas más fuerte entre 0,7 y 1,4 veces la del lote.
  Una ventana cuenta si esa separación dista al menos 7,5% de la del lote,
  es al menos el doble de fuerte y es un patrón claro; se avisa si son al
  menos 3 ventanas y el 10% del lote. Visto: un frutal con tres cuadros
  (uno a 4,5 m en un lote a 5,1 m), 20% de las ventanas; en los frutales y
  viñedos de un solo cuadro, 0-5%. Solución: dividir el lote en un
  polígono por cuadro. (Un cuadro con la misma separación pero corrido
  media hilera respecto del resto no se detecta: la posición medida en
  ventanas deriva a lo largo de lotes de cientos de hileras.)

En los 43 lotes de prueba marca 8: en 3 había un problema real (el
frutal en seto, el lote con cuadros mezclados, el hortícola) y en 5 la
decisión dudosa había salido bien. Coloreá la capa por
`revisar` y mirá esos lotes sobre la imagen; no avisa de todo: en la
parcela de viñedo con las líneas sobre la sombra de la espaldera no hay
ninguna señal que lo distinga (ver Limitaciones).

## Instalación

### 1. Instalar la dependencia que falta: OpenCV

QGIS no trae `opencv-python` instalado por defecto. Hay que instalarlo en el
**Python que usa QGIS** (no en tu Python normal del sistema):

Usá la versión exacta de abajo (probada con QGIS 3.28 LTR y su NumPy
1.24): las versiones más nuevas de OpenCV pueden pedir NumPy 2 y
actualizar el NumPy de QGIS, que deja de funcionar.

**Windows (OSGeo4W Shell, con QGIS cerrado):**
```
python -m pip install opencv-python-headless==4.11.0.86
```
(Buscá "OSGeo4W Shell" en el menú de inicio, junto con QGIS. Si aparece
"Defaulting to user installation", está bien.)

**macOS** (si instalaste la versión LTR, la aplicación es `QGIS-LTR.app`):
```
/Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install opencv-python-headless==4.11.0.86
```

**Linux:**
```
python3 -m pip install --user opencv-python-headless==4.11.0.86
```
(si QGIS usa un python del sistema, esto alcanza; si usa un entorno propio,
revisá con `Configuración > Opciones > Sistema` en QGIS cuál es el intérprete)

Reiniciá QGIS después de instalar.

### 2. Instalar el plugin

**Opción A — Instalar desde ZIP (recomendado):**
1. En QGIS: `Complementos > Administrar e instalar complementos`
2. `Instalar desde ZIP`
3. Seleccioná `crop_row_detector.zip`
4. Activá el plugin en la lista

**Opción B — Manual:**
1. Descomprimí `crop_row_detector.zip`
2. Copiá la carpeta `crop_row_detector` dentro de tu carpeta de plugins de QGIS:
   - Windows: `C:\Users\<usuario>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - Linux/Mac: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Reiniciá QGIS y activá el plugin en el administrador de complementos

## Uso

1. Abrí la caja de herramientas de Processing (`Ctrl+Alt+T` o `Procesos > Caja de herramientas`)
2. Buscá **"Detectar hileras de cultivo"** (aparece bajo el grupo con el mismo nombre)
3. Elegí tu raster de entrada y las bandas R, G, B
4. Dejá el método en **"Perfil de proyección"** salvo que tengas un motivo
   para usar Hough, ajustá parámetros si es necesario (ver tips abajo) y
   ejecutá
5. Obtenés una capa vectorial de líneas georreferenciada. Revisá el log de
   Processing: el método de perfil informa el **rumbo** detectado (grados
   respecto del norte), la **separación entre hileras** en metros y
   cuántas hileras encontró. Si el rumbo o la separación no coinciden con
   lo que ves en la imagen, el resto del resultado tampoco va a estar bien
   (ver "Distancia aproximada entre hileras" abajo).

**Importante:** recortá el raster al lote de cultivo que te interesa antes
de correr el algoritmo (`Raster > Extracción > Recortar raster por capa de
máscara`). No es obligatorio, pero corres sobre una escena más chica y
limpia es mucho más fácil de calibrar y de revisar visualmente.

## Ajuste de parámetros (tips)

### Método de perfil (default)

En general funciona con los valores por defecto. Parámetros que lo afectan:

- **Distancia aproximada entre hileras** (metros, default -1 = automático):
  solo hace falta si el log muestra un rumbo o una separación incorrectos.
  Puede pasar si otra periodicidad de la imagen es más fuerte que la de las
  hileras — por ejemplo, la distancia entre plantas *a lo largo* de la
  hilera en un frutal joven con plantas muy marcadas, que daría un rumbo
  girado 90°. Con un valor aproximado (±40%) alcanza: acota la búsqueda del
  pico espectral a ese rango.
- **Longitud mínima de línea / tramo de hilera** (px, default 50): tramos
  más cortos se descartan.
- **Espacio longitudinal máximo** (px, default 60): huecos a lo largo de la
  hilera (plantas faltantes) de hasta este largo se unen; huecos más largos
  (un camino, una cabecera) cortan la hilera en dos tramos, que conservan
  el mismo número de `hilera`.
- **Contorno de la parcela** y **Extender hasta el contorno** (m, default
  0): ver "Ajuste al contorno de la parcela" arriba.
- **Sensibilidad** (en "Parámetros avanzados", default 3): umbral de
  contraste hilera/entrehilera, en múltiplos del ruido, para *sembrar* un
  tramo (la extensión posterior usa 1 × ruido y la continuidad del patrón
  de brillo). Desde la v0.7 las zonas de menor vigor se recuperan con el
  default; bajarlo rara vez hace falta. Por debajo de ~1,5 el ruido empieza
  a generar tramos espurios (en imágenes sintéticas de ruido puro: 0
  tramos con 3, 3 con 2, 41 con 1,5).
- **Campo de distancia entre hileras por lote** (opcional): ver "Varios
  lotes en el mismo raster".
- **Franja de la hilera** (en "Parámetros avanzados", default
  Automático): qué franja es la hilera. Automático reconoce los frutales
  de copa cerrada (ver "Franja de la hilera: copa cerrada"; el log lo
  informa). *Franja más verde* fuerza el criterio de viñedos y frutales con
  suelo entre hileras (el verdor cromático, sin pasar nunca al brillo);
  *Copas iluminadas* fuerza el de copa cerrada; *Franja oscura* (desde la
  v0.17) fuerza el brillo, para canopias poco verdes o rojizas sobre
  suelo claro. Si en un frutal con pasto entre hileras las líneas caen
  sobre el pasto, elegí *Franja más verde*; si caen sobre la franja clara
  y las plantas son las oscuras, *Franja oscura*.
- **Índice de vegetación** (en "Parámetros avanzados"): ExG cromático
  (default) o ExG sin normalizar (el de las versiones ≤ 0.6). Verificá en
  el resultado que las líneas caigan sobre la canopia: si en tu imagen la
  canopia es más clara y menos verde que la entrehilera (raro), probá el
  otro.

Limitación: sin polígonos de lote, supone **un único lote** (una
orientación y una alineación de hileras) en todo el raster. Si la escena
tiene varios lotes, dibujá un polígono por lote en la capa de contorno
(ver "Varios lotes en el mismo raster").

### Método Canny + Hough

Sus parámetros específicos están en "Parámetros avanzados" del diálogo,
marcados con `[Hough]`.

- **Umbral de binarización = -1**: dejalo así para que use Otsu automático.
  Si el resultado incluye mucho suelo o le falta vegetación, poné un valor
  manual entre 0 y 255 mirando el histograma.
- **Hileras muy fragmentadas** (muchos segmentos cortos en vez de líneas
  largas): subí `Distancia máxima entre segmentos` (max_line_gap).
- **Demasiadas líneas espurias** (ruido, bordes de parcela, caminos): subí
  `Umbral de Hough` y/o `Longitud mínima de línea`.
- **Pocas o ninguna línea detectada**: bajá esos mismos parámetros.

#### Viñedo / frutal con Hough (plantas individuales, no cultivo de cobertura continua)

Canny+Hough fue pensado originalmente para una franja de vegetación
continua (cultivos de cobertura). En viñedo o frutal, cada planta es un
blob separado, y Canny detecta el contorno de **cada planta por separado**
— esos bordes salen en todas direcciones, no solo a lo largo de la hilera.
Sin corrección, esto produce una maraña de líneas cruzadas en vez de
hileras limpias. Dos parámetros para esto:

- **Cierre morfológico antes de Canny** (default 0 = desactivado): probá
  3-7 px. Une plantas cercanas en blobs más continuos antes de que Canny
  las fragmente en contornos individuales. Si tus plantas están muy
  separadas entre sí, esto no va a alcanzar por sí solo.
- **Filtrar por dirección dominante** (default: activado, tolerancia 15°):
  detecta automáticamente la orientación predominante entre todos los
  segmentos, ponderando por longitud (así los bordes reales de hilera,
  aunque sean minoría, pesan más que el ruido corto y disperso de los
  contornos de planta), y descarta los segmentos que no estén cerca de esa
  dirección — antes de fusionar. En el log de Processing vas a ver el
  ángulo detectado y una "concentración" (0 a 1): si sale muy baja (menos
  de ~0.15), significa que no hay una dirección claramente dominante — el
  ángulo detectado puede no ser el real, y conviene revisar el resultado
  a ojo, o combinar esto con el cierre morfológico y/o bajar el umbral de
  Hough para que aparezcan más bordes reales de hilera.

#### Fusión de segmentos (Hough)

Si querés una sola línea por hilera (recomendado), dejala activada.
Parámetros que la controlan:

- `Tolerancia de ángulo` (default 5°): qué tan parecida tiene que ser la
  orientación de dos segmentos para considerarlos parte de la misma hilera.
- `Distancia perpendicular máxima` (**default -1 = automático**): qué tan
  cerca (de costado, no a lo largo) tienen que estar dos segmentos para
  fusionarse. El automático mide la separación real entre hileras a
  partir de los segmentos detectados (autocorrelación de su posición
  perpendicular a la dirección dominante) y usa el 40% de esa distancia
  — pensado específicamente para viñedo/frutal de alta densidad, donde
  un valor fijo grande termina fusionando varias hileras vecinas en una
  sola. El log de Processing muestra la separación estimada y la
  tolerancia calculada. Si tus hileras son muy onduladas (no rectas), el
  automático puede ser demasiado estricto y fragmentar una misma hilera
  en varios pedazos — en ese caso poné un valor manual más alto. Si no
  se puede estimar (pocos segmentos), cae a 30 px con un aviso.
- `Espacio longitudinal máximo` (**default 60 px — la salvaguarda más
  importante**): espacio máximo, medido A LO LARGO de la hilera (no de
  costado), entre el tramo ya fusionado y el próximo segmento candidato.
  Sin este chequeo, dos segmentos con ángulo y distancia perpendicular
  parecidos se fusionan aunque estén a cientos de píxeles de distancia
  entre sí a lo largo de la hilera — con un camino, un límite de parcela,
  o cualquier zona sin relación en el medio — **incluso en un campo con
  una única dirección real consistente en toda la escena**. Es la causa
  más común de líneas absurdamente largas que cruzan zonas sin relación
  entre sí. Subilo solo si tus hileras tienen huecos reales (plantas
  faltantes) más largos que el default.
- `Longitud máxima de fusión` (default -1 = sin límite, en metros): tope
  duro adicional a cuánto puede "crecer" una hilera fusionada en total.
  Ya no suele hacer falta si el espacio longitudinal máximo está bien
  calibrado, pero sirve como límite absoluto extra si conocés la longitud
  real máxima de tus hileras.

## % de cobertura de suelo y conteo de plantas (opcionales)

Además de las líneas de hilera, el algoritmo puede calcular dos cosas más
por hilera (se activan con checkboxes, están desactivadas por default):

### % de cobertura (fracción cultivable)

Mide qué porcentaje de una franja alrededor de cada hilera está cubierto
por vegetación, **excluyendo las calles/pasillos entre hileras** (que no
deberían contarse como "suelo que debería estar cubierto").

El ancho de esa franja se define con **"Ancho de franja cultivable"**:
- **-1 (automático)**: usa la mitad de la distancia a la hilera vecina
  (con el método de perfil, la separación medida por FFT). No sabe
  distinguir una calle de tránsito ancha de una zona de cultivo
  normal, así que si tenés calles bien definidas (por ejemplo, cada tantas
  hileras pasa el tractor), el automático las va a incluir de más en la
  franja "cultivable" y el % te va a salir más bajo de lo real.
- **Manual (metros)**: si conocés el ancho real de la franja de cultivo
  (separación entre surcos menos el ancho de calle), poné ese valor. Da un
  resultado más preciso, pero hay que ajustarlo si cambia de lote en lote.

El resultado queda en el campo **`cobertura_pct`** de la capa de salida,
uno por hilera.

### Conteo de plantas (aproximado)

Cuenta manchas de vegetación separadas (componentes conectados) dentro de
esa misma franja. **Importante:** esto es confiable solo cuando las
plantas están separadas entre sí (etapas tempranas, con suelo visible
entre plantas). Si el dosel ya está cerrado (hojas de plantas vecinas
tocándose), varias plantas se funden en una sola mancha y el conteo
**subestima** el número real.

Para ayudarte a notar esto, se agrega el campo **`area_prom_planta_m2`**
(área promedio de las manchas contadas en esa hilera): si te da un valor
mucho más grande de lo que esperás para una sola planta de tu cultivo, es
señal de que el dosel está cerrado y el número de `num_plantas_est` no es
confiable en esa hilera. El parámetro **"Área mínima para contar como
planta"** filtra manchas chiquitas (ruido, restos de maleza) — subilo si
te está contando basura como planta.

Resultado en los campos **`num_plantas_est`** y **`area_prom_planta_m2`**,
uno por hilera.

## Limitaciones a tener en cuenta

- Funciona mejor con vegetación en fila claramente diferenciada del suelo
  (etapas de crecimiento intermedias). En cultivos muy jóvenes o con dosel
  cerrado, el contraste ExG puede no ser suficiente.
- Los dos métodos asumen hileras aproximadamente rectas. El método de
  perfil tolera una deriva leve (hasta ~media separación entre hileras a
  lo largo de toda la hilera) y curvas suaves que se apartan hasta 0,3
  separaciones de la recta (seguimiento lateral, paso 9), pero la salida
  es una recta por tramo: en esas hileras la línea puede quedar a
  ~0,2-0,3 separaciones de la hilera en algún punto. Supone una única
  orientación por lote; con hileras curvas no va a funcionar bien, y con
  varios lotes en la misma escena hace falta un polígono por lote.
- **El complemento nunca dibuja fuera del lote.** Si el borde de cabecera
  del polígono pasa antes de la última planta, la hilera termina en ese
  borde (visto: cabeceras que quedaron 2-3 m adentro de lo plantado en
  varios lotes de la finca de prueba). Dibujá la cabecera sobre la calle
  (puede llegar hasta su eje): el complemento ubica solo dónde termina
  cada hilera.
- Cuando las hileras se ubican por brillo (lotes con canopia poco verde),
  la línea marca el centro de la franja oscura, que incluye la sombra de
  la canopia: puede quedar corrida unos decímetros hacia el lado de la
  sombra respecto del eje de las plantas.
- El raster tiene que estar en un CRS proyectado en metros (p. ej. UTM
  19S, EPSG:32719, en Mendoza). Los recortes exportados de mapas web
  suelen venir en EPSG:4326 (grados), con píxeles que en el terreno no son
  cuadrados: reproyectalos antes (`Raster > Proyecciones > Combar
  (reproyectar)`). En Hough, la fusión por PCA da
  una sola línea recta por hilera, que no sigue la curvatura (se puede
  desactivar la fusión y trabajar con los segmentos crudos).
- Si un árbol u otra vegetación aislada cae justo sobre la prolongación de
  una hilera y es más largo (a lo largo de la hilera) que la 'Longitud
  mínima', el método de perfil puede extender la hilera hasta él. Los
  fragmentos más cortos se descartan.
- 'Extender hasta el contorno' lleva las hileras hasta el borde del
  polígono que le des: si ese polígono incluye calles de cabecera, las
  líneas las atraviesan. Por eso el default es 0.
- Unas pocas plantas chicas sueltas (menos de 3 separaciones entre
  hileras de largo), sobre una hilera que no se detectó en el resto de su
  largo, no se marcan: sin un tramo al lado no se distinguen de manchas de
  maleza o arbustos. Visto en una esquina de un frutal: tres árboles
  chicos antes de la calle.
- Si el verdor de una hilera tiene su máximo corrido respecto de la
  canopia (p. ej. pasto verde pegado a un solo lado), la línea puede quedar
  corrida hasta ~1/4 de la separación, y donde llega a eso el tramo se
  pierde. En la finca de prueba pasó en 1 hilera de ~3500.
- La alineación de extremos supone que el borde de lo plantado es
  continuo a escala de unas pocas hileras. Si una hilera realmente termina
  1-6 m antes o después que sus vecinas (p. ej. plantas arrancadas en la
  punta de una sola hilera), se la alinea igual.
- **Hileras finas** (hortalizas, < 1 m): con imágenes de ~0,2 m el
  complemento encuentra rumbo y separación, pero cada hilera sola apenas
  supera el ruido. Las marca en **modo patrón** (ver 8c): posición
  correcta, extremos con precisión de ~10 separaciones y sin fallas de
  plantas. En zonas del lote donde las plantas están menos desarrolladas el
  patrón no alcanza y quedan sin marcar (visto: el quinto oeste de una
  hortaliza de 1,5 ha). Para detalle hilera por hilera hace falta una
  imagen de drone.
- **Hortícola en camellones.** El complemento encuentra rumbo y separación
  de los camellones, pero las líneas salen cortadas: en la imagen el
  camellón y el surco casi no se distinguen en el verdor (en un lote de
  camellones cada 5,1 m, la franja a media separación tenía 93% del verdor
  del centro), y lo que marca el surco es una línea angosta de sombra y
  luz al costado. Visto: 17% del largo de los camellones marcado.
- **Copa cerrada con cubierta verde.** El criterio automático de copa
  cerrada (≥ 3,5 m entre hileras y la franja clara también vegetación)
  puede confundir un frutal joven con pasto verde entre hileras con uno de
  copa cerrada. Con árboles sueltos en marco se distingue (ver "Pasto
  entre árboles sueltos"); si los árboles no se ven sueltos (copas que se
  tocan a lo largo de la hilera), no. El log avisa cuando aplica copa
  cerrada; en ese caso, forzá *Franja más verde*.
- **Marco real (cuadrado).** Con la misma distancia entre hileras y entre
  plantas, las dos direcciones son equivalentes en la imagen y el
  complemento toma cualquiera de las dos. Indicar la distancia no alcanza
  para elegir; revisá el rumbo en el log (con menos de 1,3 veces de
  diferencia el lote sale marcado `rumbo` en el campo `revisar`).
- **Copa muy clara con su sombra negra al costado.** En un viñedo en
  espaldera con sol de costado (vid clara, tostada, y una sombra muy
  oscura y verdosa pegada), el verdor cromático y el brillo marcan los dos
  la sombra, y la línea va sobre ella (visto: parcela de viñedo con las
  líneas sobre la sombra de la espaldera). Ninguna señal del lote lo
  distingue y no se avisa. Probado sin éxito: mover la línea solo si el
  centro cae del lado claro (en un frutal en seto real empeoraba). Hace
  falta una referencia de troncos o de campo.
- **Surco o franja oscura en el medio de la entrehilera.** La entrehilera
  se mide justo a media separación. Si ahí hay algo tan parecido a la
  hilera como la hilera misma (un surco de riego oscuro, una franja de
  maleza), la hilera no se ve y la línea se corta (visto: olivar, 383 m de
  puntas en la zona donde el surco es más oscuro). Medirla en una franja
  más ancha lo arreglaba ahí pero empeoraba otros lotes (ver 0.18).
- **Cortina parecida al cultivo.** Una cortina o fila de árboles junto al
  lote que el perfil ubica como una hilera más se descarta solo si es
  mucho más marcada que las hileras interiores (ver "Nivel"). Si se parece
  a ellas, queda como una hilera más: dejala fuera del polígono del lote.
- En frutales de copa cerrada, un último árbol más pálido o sombreado que
  el resto casi no tiene verde absoluto y la línea puede terminar un árbol
  antes (2-4 m; visto en 5 de 16 hileras de un frutal).
- En los frutales con suelo entre hileras, la línea sigue la franja más
  verde (el lado sombreado de la copa más su sombra): queda ~0,1-0,15
  separaciones (0,6-0,9 m a 6 m entre hileras) hacia el lado de la sombra
  respecto del máximo de verde de la copa iluminada.
- Está pensado para ortomosaicos ya georreferenciados (GeoTIFF con
  geotransform válido), y en un **CRS proyectado (metros)**. Si el raster
  está en un CRS geográfico (grados), el % de cobertura, el conteo de
  plantas y la 'Longitud máxima de fusión' en metros van a salir mal —
  reproyectá el raster antes de usar esas opciones.
- El conteo de plantas es una **aproximación**, no un conteo exacto:
  confiá en él solo cuando el campo `area_prom_planta_m2` te dé un valor
  razonable para una sola planta de tu cultivo (ver sección de arriba).
- Necesita al menos 3 bandas (RGB) de resolución centimétrica/decimétrica
  (drone). No sirve con una sola banda satelital (ej. Sentinel-2 B08) ni,
  en general, con resoluciones de varios metros por píxel: a esa escala
  varias hileras caben dentro de un solo píxel y no hay forma de
  distinguirlas, sea cual sea el ajuste de parámetros.

## Historial de versiones

- **0.2**: validación de bandas de entrada (mensaje claro en vez de
  traceback si el raster no tiene 3+ bandas o se pide un índice de banda
  inexistente) + tope de longitud máxima de fusión (`MERGE_MAX_LENGTH`).
- **0.3**: cierre morfológico opcional antes de Canny + filtro de
  dirección dominante antes de fusionar (pensado para viñedo/frutal,
  donde Canny detecta el contorno de cada planta individual en vez de un
  borde de hilera continuo).
- **0.4**: `Distancia perpendicular máxima` de fusión pasa a ser
  automática por default (antes 30px fijo) — se auto-calibra al 40% de la
  separación real entre hileras, medida por autocorrelación. Corrige que
  hileras muy juntas (viñedo/frutal de alta densidad) colapsaran varias
  hileras vecinas en una sola.
- **0.5**: chequeo de **hueco longitudinal máximo** en la fusión
  (`MERGE_MAX_GAP`, default 60px) — la corrección más importante hasta
  ahora. Corrige que dos segmentos con ángulo y offset perpendicular
  parecidos se fusionaran aunque estuvieran a cientos de píxeles de
  distancia entre sí a lo largo de la hilera (con un camino o zona sin
  relación en el medio), incluso en escenas con una única dirección real
  consistente. Esta era la causa de las líneas absurdamente largas
  cruzando escenas grandes que las versiones 0.2-0.4 no resolvían.
- **0.6**: nuevo método de detección por **perfil de proyección orientado
  por FFT**, que pasa a ser el default (el de Canny + Hough queda como
  alternativa, con sus parámetros en "Parámetros avanzados"). Probando la
  0.5 sobre un raster real de viñedo se vio que el problema de fondo no
  era la fusión sino la detección: con hileras angostas y juntas, Hough
  encuentra sobre todo rectas que *cruzan* las hileras, y la dirección
  dominante sale perpendicular a la real (ver "Por qué dejó de ser el
  default" arriba). También: los píxeles transparentes/nodata se excluyen
  del Otsu y de todo el cálculo (antes el área fuera de un recorte, con
  RGB = 0, corría el umbral), las coordenadas de salida se toman en el
  centro del píxel (antes había un corrimiento de medio píxel), y la capa
  de salida suma el campo `hilera` con el método de perfil.
- **0.7**: dos correcciones sobre el mismo raster real.
  1. **Las líneas caían sobre la entrehilera.** El ExG sin normalizar
     (2G − R − B) escala con el brillo, y en esa imagen la canopia es la
     franja *oscura*: la entrehilera clara con pasto daba más ExG que la
     canopia. Ahora el índice es el **ExG cromático** (2G − R − B)/(R + G +
     B) de Woebbecke et al. (1995), sobre bandas suavizadas σ = 1 px. Sobre
     el raster de prueba, el verdor cromático medio sobre las líneas pasó a
     0,65 contra 0,37 en la entrehilera. Consecuencia: la cobertura de la
     v0.6 en ese raster (~40%) medía la entrehilera; ahora mide canopia
     (~24%).
  2. **Inicios y finales irregulares.** Umbral con histéresis y extremos
     localizados a media altura; continuidad de la hilera por patrón de
     brillo donde el verdor se debilita o se invierte; hileras de borde
     agregadas por continuidad de la grilla; fragmentos terminales cortos
     descartados; y parámetros nuevos **Contorno de la parcela** (recorte
     estricto, ambos métodos) y **Extender hasta el contorno**. En el
     raster de prueba: 172 hileras, una línea continua cada una.
- **0.8**: las hileras ya no entran en la calle de cabecera. En la 0.7,
  'Extender hasta el contorno' valía 15 m por defecto y el contorno del
  lote incluye las calles, así que las líneas las atravesaban hasta el
  borde (~1,5 km de más en el raster de prueba: 24,4 km contra 22,9 km).
  Ahora el default es 0 y los extremos que se apartan del borde que forman
  las hileras vecinas se alinean con él (ver "Extremos en el borde de lo
  plantado"). En el raster de prueba, todos los extremos quedan a 3,5-5,6
  m del contorno del lote (el ancho de la calle), y los de las hileras del
  rincón con árboles no se tocan.
- **0.9**: **procesamiento por lote**. Con varios polígonos en la capa de
  contorno, cada lote se procesa por separado (orientación, separación,
  alineación de hileras y numeración propias; campo `lote`). Motivo: al
  aplicarlo a dos recortes con varios lotes, uno con monte, frutal y
  viñedos con distintas orientaciones falló (la FFT eligió una
  periodicidad que no era de ninguno) y el otro, con 6-7 lotes de viñedo,
  cruzaba las calles entre lotes y desalineaba las hileras de un lote
  ubicado detrás de otro. En una escena sintética con esos mismos
  problemas: con contorno único, cero hileras en dos de los tres lotes;
  por lote, el 100% de las líneas sobre la canopia en los tres, ninguna
  cruzando la calle.
- **0.10**: probado sobre una finca completa (103 ha, 19 lotes: 16 de
  viñedo y 3 de frutal, más un sector con casas y arroyo). Dos
  correcciones, ambas por el patrón de brillo (ver "El brillo cumple además
  dos funciones de control"):
  1. En 4 lotes con la canopia casi sin verdor en la imagen, las hileras
     salían fragmentadas (17-39% de completitud): ahora se ubican por
     brillo (98-100%).
  2. En franjas de 3 lotes con la entrehilera más verde que la hilera, 15
     líneas caían sobre la entrehilera: ahora se verifican contra el
     brillo y se reubican (0 líneas sobre la entrehilera).

  Resultado en la finca: 2824 líneas, 367 km de hileras, completitud de
  92-100% en los 19 lotes, rumbo y separación propios por lote (viñedo 22°
  y 2,0 m; frutal 25,3° y 5,95 m), en ~70 s para un raster de 46 Mpx.
- **0.11**: **estabilidad ante el remuestreo**. Al recortar la misma
  imagen de Google con un polígono de la finca redibujado (otro origen de
  la grilla, mismo tamaño de píxel), un lote pasó de 99% a 86% de
  completitud. Dos causas, las dos por umbrales discontinuos:
  1. La banda central tenía 2·round(T/4)+1 columnas: con hileras a 2 m y
     píxel de 0,2 m (T ≈ 10 px), una separación estimada de 9,9 o de 10,1
     px daba 5 o 7 columnas, y el ruido estimado cambiaba un 28%. Ahora el
     ancho es continuo en T.
  2. El umbral del brillo salía de la cola negativa de su contraste, que
     en las hileras bien vistas tiene menos del 0,1% de las muestras: por
     debajo de 50 muestras se usaba un respaldo (10% del contraste típico)
     y por encima, el valor cuadrático medio de un puñado de anomalías
     (en ese lote, una sola hilera aportaba el 94-99%). El umbral pasó de
     22 a 82, casi el contraste típico. Ahora es siempre relativo al
     contraste típico.

  Resultado: los dos recortes dan la misma completitud por lote (±0,5%)
  y el lote afectado vuelve a 99%. Con el polígono redibujado (110 ha):
  2802 líneas, 371 km, completitud de 91-100% en los 19 lotes y ninguna
  línea sobre la entrehilera.
- **0.12**: **hileras sin marcar**, a partir de la revisión visual de la
  finca. Tres correcciones (ver "Hileras de borde", "Planta por planta" y
  "Grilla del marco de plantación"):
  1. Hileras de borde: donde el borde del lote pasaba sobre la primera o
     la última hilera, esa hilera se perdía entera (en un frutal, una fila
     completa de árboles).
  2. Plantas chicas en la punta de la hilera (replantes, menor vigor): la
     línea terminaba en la última planta grande.
  3. Hileras de plantas chicas o ralas que no forman un máximo en el perfil
     del lote: ahora se prueban todas las posiciones del marco de
     plantación.
  4. Nivel de los tramos (ver "Nivel"): con lotes que toman parte de la
     calle aparecían líneas sobre huellas de tractor (12 en la finca);
     ahora se descartan.

  Resultado en la finca (110 ha, ahora 21 lotes: se agregaron dos
  cuarteles chicos de viñedo entre los brazos del arroyo, y los polígonos
  de los lotes se ampliaron hasta el eje de las calles, porque en las
  cabeceras cortaban las puntas de las hileras y a los costados las
  hileras de borde): 2887 líneas, 383 km, ninguna línea sobre la
  entrehilera ni sobre la calle.

  **Recomendación:** el contorno de cada lote puede tomar parte de la
  calle (hasta su eje); no hace falta dibujarlo ajustado a las plantas.
  Lo que sí conviene dejar afuera son las cortinas rompevientos y los
  árboles sueltos paralelos a las hileras: son vegetación en fila y se
  detectarían como una hilera más. La canopia que queda sin
  línea bajó en todos los lotes (en los frutales, de 8-10% a 4-9%; en los
  viñedos, de 1-3,7% a 0,5-3,3%, donde lo que queda es sobre todo sombra
  fuera de la franja de la hilera y extremos de 1-2 plantas).
- **0.13**: **otros cultivos**. Probado sobre dos parcelas nuevas (un
  frutal de copa cerrada a 4,5 m y una hortaliza en líneas a 0,85 m), con
  la condición de no empeorar la finca de 21 lotes.
  1. **Frutal de copa cerrada: las líneas caían entre copas**, sobre la
     sombra (ver "Franja de la hilera: copa cerrada"). Además, en ese lote
     5 hileras terminaban 12-17 m antes y 2 no se marcaban. Ahora: 16
     hileras sobre las copas (separaciones de 4,3-4,7 m; antes alternaban
     4,2 y 4,7), las puntas norte en el primer árbol y las sur en el último
     (salvo 5 hileras que terminan un árbol antes, donde el último es más
     pálido), sin el cerco del borde.
  2. **Separación automática en hileras finas**: espectro blanqueado (ver
     el paso 4). Antes tomaba un manchón de 36 m; ahora 0,85 m sin
     indicarla.
  3. **Un polígono solo se procesa como lote**: rumbo y separación
     estimados solo adentro (ver "Varios lotes en el mismo raster").
  4. **Distancia entre hileras por lote** (campo de la capa de lotes).
  5. **Hileras finas**: menos suavizado y modo patrón (ver 8c). En la
     hortaliza: 9,3 km de líneas sobre las hileras (antes, 2 líneas
     falsas).
  6. **Hileras de borde en el marco de plantación** (ver "Hileras de
     borde").

  En la finca de 21 lotes el resultado es igual al de la v0.12 en 20 lotes
  (misma completitud, ninguna línea sobre la entrehilera ni sobre la
  calle); en el lote 6 se descartó un tramo de 10 m a 3,66 separaciones de
  la hilera más cercana, fuera del marco. Total: 2886 líneas, 383,2 km. Test
  nuevo `tests/test_v013.py` (escena sintética con copa cerrada, hileras
  finas sobre manchones, viñedo de control, campo de separación y polígono
  único).
- **0.14**: **hileras incompletas** (revisión visual de 5 lotes de vid de
  la finca de prueba: hileras bien visibles con tramos sin vector). Tres
  causas:
  1. **Hileras levemente torcidas** (ver "Seguimiento lateral", paso 9):
     donde la hilera real se apartaba ~1/4 de separación de su recta, el
     contraste se invertía y el tramo se perdía (hasta 45 m en una
     hilera). También quedaban huecos donde una franja oscura (maleza, una
     acequia) cruza las hileras en diagonal.
  2. **Puntas separadas por una planta débil** (ver "Extremos"): un corte
     de 0,4 m a 10 m de la punta hacía descartar esos 10 m como si fueran
     un árbol pasando la cabecera. Ahora esa punta se suma si no pasa del
     borde que forman las hileras vecinas.
  3. **Polígonos que cortaban la cabecera**: en varios lotes el borde de
     cabecera pasaba 2-3 m antes de la última planta, y las líneas nunca
     salen del lote. No es del algoritmo sino de la capa de lotes: se
     estiraron esas cabeceras hasta la última planta + 1,5 m
     (`tests/extender_cabeceras.py`) y se agregó el aviso en la ayuda y en
     las limitaciones.

  En los 5 lotes revisados, la canopia sin vector al final de las hileras
  bajó de 311 m a 100 m (47 → 12, 22 → 14, 78 → 17, 65 → 24 y 99 → 33 m),
  y ninguna cabecera corta hileras. En la finca de 21 lotes: 2875 líneas,
  384,0 km (+0,8 km), ninguna línea nueva sobre la calle ni sobre la
  entrehilera. En frutales el seguimiento casi no actúa (0,6 m son 0,1
  separaciones y la copa es ancha): el frutal del lote 4 queda igual y el
  del lote 7 suma 46 m, hileras que ahora llegan al último árbol. En la
  finca de la hortaliza y el frutal de copa cerrada el resultado es igual
  al de la v0.13. Test nuevo `tests/test_v014.py` (hileras con una panza
  de 0,3 separaciones, punta tras dos plantas faltantes y árbol en la
  cabecera, que no se une): la v0.13 falla los dos primeros casos.
- **0.15**: **revisión hilera por hilera** de la finca de 21 lotes con una
  auditoría automática (`tests/auditar_hileras.py`: para cada hilera, y
  para cada posición del marco sin línea, compara metro a metro la
  canopia de la imagen con las líneas, y arma recortes de cada falla con
  `tests/ver_fallas.py`). Fallas corregidas, las dos en **hileras de
  borde**:
  1. **Plantas chicas o ralas descartadas por nivel** (ver "Nivel"): en los
     frutales, las hileras de árboles chicos del borde quedaban sin línea o
     cortas (lote 13: la hilera junto a la cortina; lote 4: un tramo de 33 m
     de la hilera de borde). Al probarlo apareció el caso contrario: unos
     arbustos claros sueltos sobre un sendero, al borde del lote 4, también
     pasaban; por eso el rescate exige que el tramo sea más oscuro que la
     entrehilera.
  2. **Hileras de borde corridas** (ver 8b): la posición que se probaba se
     afinaba al máximo del perfil de todo el lote y quedaba a 0,2-0,4 T de
     la hilera; se medía de costado y la hilera de borde se perdía (dos
     viñedos: la primera hilera junto a la calle).

  Falsas alarmas de la auditoría que no son fallas: filas de árboles de
  las calles, cercos y construcciones dentro del polígono (no son
  hileras), y huecos de plantas faltantes de hasta 12 m que la línea
  cruza a propósito (`Espacio longitudinal máximo`).
  Pendiente: en los frutales con suelo entre hileras la línea sigue
  corrida hacia el lado de la sombra (~0,1-0,15 T, ver Limitaciones);
  para corregirla hace falta una referencia medida de la posición de los
  troncos. Test nuevo `tests/test_v015.py`.
- **0.16**: **frutal en marco rectangular** (parcela de prueba nueva: frutal
  joven de 1,5 ha, 5 × 4 m, pasto entre hileras). Dos errores encadenados:
  1. **Líneas perpendiculares a las hileras**: se tomaba la alineación
     cruzada de los árboles (4,02 m a 115°) en vez de las hileras (4,97 m
     a 30°). Ver "Marco de plantación" en el paso 4. Indicar la distancia
     entre hileras tampoco lo corregía: ahora se respeta.
  2. **Con la dirección bien, las líneas iban sobre el pasto**: el pasto
     verde entre hileras activaba copa cerrada. Ver "Pasto entre árboles
     sueltos" en el paso 5.

  Resultado en esa parcela: rumbo 30,0°, separación 5,01 m, líneas sobre
  los árboles en toda la parcela (antes, 38 tramos cruzados y la zona con
  suelo trabajado sin marcar). Pendiente ahí: 3 hileras sin línea y
  algunas cortas en la zona con suelo trabajado. En la finca de 21 lotes,
  la parcela de viñedo 400 y las dos de Finca, el resultado es idéntico al
  de la v0.15 (ningún lote muestra marco). Test nuevo `tests/test_v016.py`
  (la v0.15 falla 6 de sus 12 chequeos).
- **0.17**: **evaluación ciega**. Otra sesión, sin acceso a este trabajo,
  probó la v0.16 como un usuario nuevo en 6 lotes que el complemento
  nunca había visto (dos viñedos, un viñedo de franjas rojizas, un
  olivar, un frutal en seto y un frutal en marco): con los valores por
  defecto, 3 bien y 3 mal, y el log sin ningún aviso en los 3 malos.
  Cambios:
  1. **Olivar girado 90°** (127°, 7,8 m en vez de 36°, 12,25 m): marco
     entre índices (paso 4). Además, la distancia indicada no se
     respetaba (el borde de la ventana de búsqueda contaba como pico, y
     luego un pico débil en otra dirección quedaba más cerca del valor):
     ahora, el pico más fuerte a ±10%. La re-estimación a resolución
     reducida ya no puede irse a otro patrón.
  2. **Franja**: la franja oscura se usa si la canopia no es verde (lote
     de franjas rojizas: las líneas iban sobre la franja clara y sobre
     calles), y no se usa si es la sombra al costado de una canopia verde
     (frutal en seto: de 27 a 57 de 61 hileras). Nueva opción *Franja
     oscura*; *Franja más verde* ya no pasa al brillo.
  3. **Hileras cortadas** por un corte de 0,4 m (frutal en marco: dos
     hileras con 30-70 m sin línea): microcortes; y el control de nivel
     prueba las partes de un tramo antes de descartarlo entero.
  4. **Cortina** en una posición agregada del borde, y **árboles de la
     calle** como hilera extrema (Francesco los vio claramente distintos
     del cultivo en el frutal en marco; el mismo criterio saca la fila de
     árboles del borde de la parcela de viñedo 400, error conocido desde
     la v0.15): se descartan.
  5. **Aviso de revisión** y campo `revisar`.
  6. **Alineación de extremos** más tolerante con el ruido del borde (una
     hilera de viñedo 3,5 m metida en la calle de cabecera).
  7. **Olivar con hileras incompletas** (Francesco, al probar la v0.17):
     siembra por suma de rachas, continuación tras árboles faltantes y
     hueco máximo de al menos 1,25 T (ver "Árboles sueltos muy
     separados"); nivel sin las rampas de las puntas.
     Y el lote de canopia rojiza, que parecía peor que con la v0.16 (que
     cubría todo, pero sobre las franjas claras; Francesco confirmó que
     las plantas son las oscuras): el verdor siembra tramos cuando la
     hilera se ubica por la franja oscura.
  8. Detalles: el nombre del algoritmo ya no dice "(ExG + Hough)"; el
     mensaje "verdor −6432% … nan veces" ya no aparece.

  Resultado en los 6 lotes: olivar 36,2°, 12,25 m, 22 de 23 hileras;
  seto 57 de 61; franjas rojizas sobre las oscuras y sin líneas en calles;
  cortina y árboles de la calle descartados; las dos hileras cortadas,
  completas (52 → 120 m y 90 → 118 m). En el lote de franjas rojizas, la
  auditoría hilera por hilera da 2 tramos falsos (10 m) y 8 hileras sin
  marcar (0,5 km; antes 45). En la finca de 21 lotes y las parcelas de Finca
  (incluida la 500): la misma cantidad de líneas y ninguna corrida de
  costado; solo cambian puntas (la 500 suma una hilera de 47 a 105 m); en
  la parcela de viñedo 400 sale la fila de árboles del borde. Pruebas:
  `tests/regresion.py` y `tests/comparar_regresion.py` (los 31 lotes
  contra una versión anterior), `tests/test_v017.py` (la v0.16 falla la
  escena de canopia rojiza). Sigue sin resolverse la línea sobre la
  sombra en la parcela de viñedo 400 (ver Limitaciones).
- **0.18**: **evaluación contra una referencia corregida a mano**. Hasta
  la v0.17 cada cambio se juzgaba comparando una versión con otra y con
  una auditoría automática: eso detecta cambios, no errores. Francesco
  corrigió a mano las líneas de 5 lotes (olivar, frutal en seto, frutal
  en marco, frutal joven de la parcela 500 y un lote de viñedo de la
  finca): borró las falsas, estiró y recortó puntas, corrió las mal
  ubicadas y dibujó las que faltaban. `tests/crear_referencia.py` arma
  la capa para corregir y `tests/evaluar_referencia.py` puntúa cualquier
  corrida contra ella (largo de hilera real con línea encima, largo de
  línea sobre hileras reales, F1, y el desglose: sin marcar, huecos,
  puntas cortas o largas, líneas corridas o falsas). Con eso, cada
  versión desde la v0.16 sube en el total (F1 0,834 → 0,927 → 0,940).
  Un cambio entra solo si sube la referencia sin empeorar ningún lote.
  Cambio:
  1. **Frutal en seto**: con la franja oscura juzgada sombra, la
     verificación de posición llevaba 19 hileras a la sombra (1 m del
     centro de la canopia, todas del mismo lado; Francesco las corrió).
     Ahora van a la franja oscura menos el corrimiento de la sombra
     (ver "Verificación de posición"). Lote en seto: F1 0,703 → 0,725
     (líneas corridas 1413 → 915 m, falsas 172 → 11 m; quedan hileras
     que se corren hacia el otro lado y 3 más sin marcar). Solo cambia
     en lotes donde la franja oscura se juzgó sombra: en los otros 30
     lotes de prueba, idéntico.

  2. **Revisión de código** (sin cambios en los resultados de los 31
     lotes de prueba): una punta alineada con el borde de las vecinas
     podía caer fuera de la grilla y cortar el proceso con un error; el
     rescate de plantas chicas ya no puede devolver una cortina o fila de
     árboles descartada como hilera extrema; la reubicación hacia la
     canopia no manda una hilera fuera del lote; la máscara binaria (Otsu)
     solo se calcula si se usa (Hough, cobertura, conteo); el perfil a lo
     ancho y el recorte a media altura, que estaban repetidos, quedan en
     una sola función cada uno; el texto de 'Distancia entre hileras'
     describe la búsqueda real (0,7-1,4 veces, prefiriendo ±10%).

  Probado y descartado (no pasó la referencia): medir la entrehilera en
  una franja alrededor de ±T/2 en vez de una sola línea. En el olivar
  cada entrehilera tiene un surco oscuro justo en el medio, y en la zona
  norte es tan oscuro como los olivos: 383 m de puntas cortas. Con la
  franja se recuperaban, pero con el promedio la parcela 500 perdía
  hileras (allí el centro es una huella clara, la mejor referencia), y
  con el mínimo las hileras entraban en la cortina y las puntas se
  alargaban en todos los lotes. Queda como limitación (ver Limitaciones).
- **0.19**: **primera prueba con lotes nuevos**. Francesco sumó 12 lotes
  (frutales, viñedos, un frutal con cobertura muy cargada y un hortícola
  en camellones); se corrió la v0.18 sin tocar nada y él corrigió las
  líneas de los lotes enteros (1500 hileras). Medida honesta en los 9
  frutales y viñedos: **F1 0,969** en la primera corrida (5 lotes por
  encima de 0,99). Fallaron un frutal joven (líneas en el medio de la entrehilera),
  el hortícola (ninguna línea) y el lote de cobertura cargada (descartado
  por ahora: hace falta una imagen de más resolución). Cambios:
  1. **Franja en la zona gris**: con la canopia apenas verde y la franja
     oscura en otro lugar, se usa la oscura (ver "Índice principal cuando
     el verdor no sirve"). El frutal joven pasa del interfilar a los árboles.
  2. **Espectro blanqueado**: solo picos con potencia propia y la
     fundamental del más fuerte (ver "Espectro blanqueado"). El hortícola
     pasa de 0,38 m (el granulado de la imagen) a 5,13 m, la separación
     de los camellones: F1 0 → 0,26. Las líneas salen cortadas porque el
     camellón casi no se distingue del surco en el verdor (ver
     Limitaciones).

  3. **Aviso de cuadros mezclados** (motivo `cuadros` en el campo
     `revisar`, ver "Aviso de revisión"): el frutal joven juntaba en un
     solo polígono tres cuadros de plantación, y en los otros dos las
     líneas salían intercaladas o faltaban (F1 0,935 con el lote entero
     corregido). No cambia ninguna línea: avisa que conviene dividir el
     lote.
  4. **Entrehileras alternadas** (ver "Continuidad del patrón"): un
     viñedo con rastra hilera por medio perdía 0,8 km de puntas porque la
     vid no se distingue de la cobertura del lado no rastreado. F1 0,942 →
     0,968; el otro viñedo de esa finca, 0,951 → 0,952.

  Con los cuatro cambios, en las 15 referencias corregidas (1838
  hileras): F1 0,963 → 0,964 sin que baje ningún lote (y el frutal joven
  de 0,096 a 0,935). Cambian las líneas de 6 de los 43 lotes de prueba.
  Test nuevo: `tests/test_v019.py` (la v0.18 falla la escena de
  camellones con granulado).

  Probado y descartado: (a) trabajar con píxeles de hasta 0,5 m en
  separaciones grandes para ubicar mejor las puntas (en un frutal a 20 m
  se pasaban 4-6 m): el olivar y ese frutal empeoraban mucho (F1 0,93 →
  0,79 y 0,94 → 0,41), porque el método supone ~12 píxeles por hilera;
  (b) la tolerancia de un lado en todos los lotes: mejoraba el viñedo
  con rastra y la parcela 500 pero empeoraba el frutal joven, un frutal a
  7 m y un viñedo (líneas en un callejón pelado).

## Referencias

- Matas, J., Galambos, C., & Kittler, J. (2000). Robust detection of lines
  using the progressive probabilistic Hough transform. *Computer Vision and
  Image Understanding*, 78(1), 119–137.
- Otsu, N. (1979). A threshold selection method from gray-level
  histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1),
  62–66.
- Søgaard, H. T., & Olsen, H. J. (2003). Determination of crop rows by
  image analysis without segmentation. *Computers and Electronics in
  Agriculture*, 38(2), 141–158.
- Welch, P. D. (1967). The use of fast Fourier transform for the
  estimation of power spectra: a method based on time averaging over
  short, modified periodograms. *IEEE Transactions on Audio and
  Electroacoustics*, 15(2), 70–73.
- Woebbecke, D. M., Meyer, G. E., Von Bargen, K., & Mortensen, D. A.
  (1995). Color indices for weed identification under various soil,
  residue, and lighting conditions. *Transactions of the ASAE*, 38(1),
  259–269.
