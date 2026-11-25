"""
Genera las 13 hojas de actividad (7 CMB + 6 Cefeidas) a partir de las plantillas
"Datos Observacionales CMB.docx" y "Datos Observacionales Cefeidas.docx".

Cada hoja tiene datos distintos, pero TODAS se generan con el valor actual de H0 de su método:
  - CMB (Planck 2018):        H0 = 67.4 km/s/Mpc
  - Cefeidas (SH0ES 2022):    H0 = 73.0 km/s/Mpc
Como la actividad es una competencia, ninguna hoja tiene ventaja: todas tienen el mismo número
de outliers y solo el equipo que descarta TODOS sus outliers llega al valor real. Cada outlier,
por sí solo, ya desvía el ajuste.

Además genera "CONTROL profesor - Actividad H0.html": una copia del HTML de los estudiantes que
carga sola los datos de la hoja elegida y muestra la clave, para revisar los ajustes rápido.

Uso:  python3 generar_hojas.py      (desde esta carpeta)
"""
import csv
import html
import json
import re
import zipfile
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
PLANTILLAS = AQUI.parent
SEMILLA = 2026
C_LUZ = 300000.0  # km/s, igual que en el notebook de Cefeidas

H0_CMB_REAL = 67.4
H0_CEF_REAL = 73.0
N_CMB, N_CEF = 7, 6
# Filas por hoja y outliers por hoja (iguales dentro de cada método, para que la dificultad sea pareja).
# Cefeidas tiene menos filas porque cada una lleva más números que dictar: así ambas tablas toman
# un tiempo parecido en llenarse (CMB: 15 × 3 datos; Cefeidas: 10 × 4 datos).
FILAS_CMB, FILAS_CEF = 15, 10
OUTLIERS_CMB, OUTLIERS_CEF = 3, 2
# Si un equipo no descarta nada, su ajuste queda entre 4 y 10 unidades del valor real
DESVIO_MIN, DESVIO_MAX = 4, 10

HTML_ESTUDIANTES = PLANTILLAS / "Actividad H0 - Estudiantes.html"
HTML_CONTROL = AQUI / "CONTROL profesor - Actividad H0.html"

rng = np.random.default_rng(SEMILLA)


# ----------------------------------------------------------------------------
# Modelo teórico del CMB (idéntico al del notebook y al del HTML)
# ----------------------------------------------------------------------------
def modelo_teorico_cmb(x, H0):
    x_aj = x - (H0 - 67) * 1.5
    return (5000 * np.exp(-((x_aj - 220) / 120) ** 2)
            + 2500 * np.exp(-((x_aj - 540) / 100) ** 2)
            + 1500 * np.exp(-((x_aj - 800) / 80) ** 2)
            + 800 * np.exp(-x_aj / 400))


def repartidos(lo, hi, n=15):
    """n valores repartidos a lo largo de [lo, hi]: uno al azar dentro de cada tramo,
    para que los puntos no queden amontonados en una zona del gráfico."""
    ancho = (hi - lo) / n
    return lo + ancho * (np.arange(n) + rng.uniform(0.2, 0.8, n))


def escalas_cmb():
    """15 escalas angulares (múltiplos de 10) repartidas entre 150 y 1000, separadas >= 30."""
    while True:
        x = (np.round(repartidos(150, 1000) / 10) * 10).astype(int)
        if np.all(np.diff(x) >= 30):
            return x


def generar_cmb(H0):
    x = escalas_cmb()
    y_teo = modelo_teorico_cmb(x, H0)
    y = y_teo + rng.normal(0, 0.03 * y_teo + 50)
    # outliers lejos de los extremos para que se noten
    idx_out = rng.choice(np.arange(1, FILAS_CMB - 1), OUTLIERS_CMB, replace=False)
    for i in idx_out:
        if rng.random() < 0.5:
            y[i] = y_teo[i] + rng.uniform(1300, 2200)
        else:
            y[i] = y_teo[i] * rng.uniform(0.25, 0.45)
    y = np.clip(np.round(y / 10) * 10, 50, None).astype(int)

    filtro = rng.choice([100, 143, 217], FILAS_CMB)
    return dict(x=x, y=y, calidad=calidades(idx_out, FILAS_CMB), filtro=filtro, outliers=sorted(int(i) for i in idx_out))


def calidades(idx_out, n):
    calidad = rng.choice(["Excelente", "Buena", "Media"], n, p=[0.4, 0.4, 0.2]).astype(object)
    # Casi todos los outliers tienen señal "Mala"; uno puede ser "Media" para que piensen
    for k, i in enumerate(idx_out):
        calidad[i] = "Media" if (k == 0 and rng.random() < 0.35) else "Mala"
    return list(calidad)


def rms(residuos):
    """Error del ajuste tal como lo muestra el HTML (raíz del error cuadrático medio)."""
    return np.sqrt(np.mean(residuos ** 2))


def errores_cmb(h, usar):
    grilla = np.arange(40, 101)  # deslizador del HTML: enteros entre 40 y 100
    x, y = h["x"][usar], h["y"][usar]
    return grilla, np.array([rms(y - modelo_teorico_cmb(x, g)) for g in grilla])


def mejor_ajuste(grilla, err):
    return float(grilla[int(np.argmin(err))])


def minimo_visible(grilla, err, valor):
    """True si, con el error redondeado a 1 decimal como lo muestra el HTML, `valor` tiene un
    error estrictamente menor que sus dos vecinos del deslizador (no hay empate a la vista)."""
    e = np.round(err, 1)
    k = int(np.argmin(np.abs(grilla - valor)))
    return e[k] < e[k - 1] and e[k] < e[k + 1]


# ----------------------------------------------------------------------------
# Cefeidas
# ----------------------------------------------------------------------------
def generar_cef(H0):
    # distancias repartidas entre 5 y 45 Mpc, en orden aleatorio dentro de la tabla
    # Sin extinción (ya viene corregida) y con pocos decimales, para que dictar sea rápido:
    # m y M con 1 decimal, z con 4.
    d = rng.permutation(repartidos(5, 45, FILAS_CEF))
    M = np.round(rng.uniform(-6.0, -2.5, FILAS_CEF), 1)
    m = np.round(M + 5 * np.log10(d * 1e6) - 5, 1)
    z = H0 * d / C_LUZ * (1 + rng.normal(0, 0.025, FILAS_CEF))

    idx_out = rng.choice(np.arange(FILAS_CEF), OUTLIERS_CEF, replace=False)
    for k, i in enumerate(idx_out):
        if k % 2 == 0:  # velocidad peculiar enorme / mala identificación del redshift
            z[i] *= rng.choice([rng.uniform(1.45, 1.8), rng.uniform(0.4, 0.6)])
        else:  # cefeida mezclada con otra estrella: se ve más brillante
            m[i] = np.round(m[i] - rng.uniform(0.9, 1.3), 1)
    z = np.round(z, 4)
    return dict(m=m, M=M, z=z, calidad=calidades(idx_out, FILAS_CEF), outliers=sorted(int(i) for i in idx_out))


def dist_mpc(m, M):
    return 10 ** ((m - M + 5) / 5) / 1e6


def errores_cef(h, usar):
    d = dist_mpc(h["m"], h["M"])[usar]
    v = C_LUZ * h["z"][usar]
    grilla = np.arange(40, 110.01, 0.5)  # deslizador del HTML: pasos de 0,5 entre 40 y 110
    return grilla, np.array([rms(v - g * d) for g in grilla])


# ----------------------------------------------------------------------------
# Edición de la plantilla .docx
# ----------------------------------------------------------------------------
def coma(v, dec):
    return f"{v:.{dec}f}".replace(".", ",")


RPR_CELDA = '<w:rPr><w:rFonts w:ascii="Arial" w:cs="Arial" w:eastAsia="Arial" w:hAnsi="Arial"/><w:rtl w:val="0"/></w:rPr>'


def set_texto_celda(tc, texto, uniforme=False):
    """Deja una sola corrida de texto en la celda, conservando el formato de la primera.

    Con uniforme=True usa un formato neutro y centrado (la plantilla de Cefeidas trae
    colores, negritas y alineaciones distintas en cada celda). Un "\n" en el texto es un salto de línea.
    """
    if uniforme:  # bordes propios de algunas celdas de la plantilla (p. ej. una línea azul)
        tc = re.sub(r"<w:tcBorders>.*?</w:tcBorders>", "", tc, flags=re.S)
    p_ini = tc.find("<w:p ")
    if p_ini < 0:
        p_ini = tc.find("<w:p>")
    p_fin = tc.rfind("</w:p>")
    p = tc[p_ini:p_fin]
    if uniforme:  # la plantilla trae párrafos vacíos extra en algunas celdas: se deja solo el primero
        p = p[:p.find("</w:p>")] if "</w:p>" in p else p
    runs = re.findall(r"<w:r[ >].*?</w:r>", p, re.S)
    rpr = re.search(r"<w:rPr>.*?</w:rPr>", runs[0], re.S) if runs else None
    rpr = rpr.group(0) if rpr else ""
    if uniforme:
        rpr = RPR_CELDA
        p = re.sub(r'<w:jc w:val="[a-z]+"/>', '<w:jc w:val="center"/>', p)
        p = re.sub(r"<w:ind [^>]*/>", '<w:ind w:left="0" w:right="0" w:firstLine="0"/>', p)
    lineas = [f'<w:t xml:space="preserve">{html.escape(l, quote=False)}</w:t>' for l in texto.split("\n")]
    nuevo_run = f'<w:r>{rpr}{"<w:br/>".join(lineas)}</w:r>'
    sin_runs = re.sub(r"<w:r[ >].*?</w:r>", "", p, flags=re.S)
    return tc[:p_ini] + sin_runs + nuevo_run + tc[p_fin:]


def reemplazar_parrafo(xml, contiene, nuevo):
    """Reemplaza el texto del párrafo que contiene `contiene` (texto unido de sus corridas)."""
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        texto = "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p))
        if contiene in texto:
            return xml.replace(p, set_texto_celda(p, nuevo), 1)
    raise ValueError(f"No se encontró el párrafo: {contiene}")


def reemplazar_corrida(xml, viejo, nuevo):
    patron = re.compile(r"(<w:t(?: [^>]*)?>)" + re.escape(viejo) + r"(</w:t>)")
    xml, n = patron.subn(lambda mm: mm.group(1) + nuevo + mm.group(2), xml, count=1)
    if n == 0:
        raise ValueError(f"No se encontró la corrida: {viejo}")
    return xml


def rellenar_tabla(xml, filas):
    tablas = [m.start() for m in re.finditer("<w:tbl>", xml)]
    t_ini = tablas[1]
    t_fin = xml.find("</w:tbl>", t_ini)
    tabla = xml[t_ini:t_fin]
    trs = re.findall(r"<w:tr[ >].*?</w:tr>", tabla, re.S)
    assert len(trs) == 16, len(trs)
    for tr in trs[1 + len(filas):]:  # la plantilla trae 15 filas; se quitan las que sobran
        tabla = tabla.replace(tr, "", 1)
    for tr, valores in zip(trs[1:], filas):
        tcs = re.findall(r"<w:tc>.*?</w:tc>", tr, re.S)
        assert len(tcs) == len(valores), (len(tcs), valores)
        nuevo_tr = re.sub(r"<w:trHeight [^>]*/>", "", tr)  # la última fila de la plantilla es más alta
        for tc, v in zip(tcs, valores):
            nuevo_tr = nuevo_tr.replace(tc, set_texto_celda(tc, str(v), uniforme=True), 1)
        tabla = tabla.replace(tr, nuevo_tr, 1)
    return xml[:t_ini] + tabla + xml[t_fin:]


# Créditos que van en el pie de página de todas las hojas (primera línea en negrita)
CREDITOS = [
    "Departamento de Astronomía - Universidad de Concepción",
    "Encargadas: Mary Loli Martínez - Bárbara Iraira",
    "Estudiantes: Felipe Cubillos - Diego González",
]
RID_PIE = "rIdPieCreditos"


def xml_pie_pagina():
    parrafos = []
    for i, linea in enumerate(CREDITOS):
        negrita = "<w:b/><w:bCs/>" if i == 0 else ""
        borde = ('<w:pBdr><w:top w:val="single" w:sz="4" w:space="4" w:color="999999"/></w:pBdr>'
                 if i == 0 else "")
        parrafos.append(
            f'<w:p><w:pPr>{borde}<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
            f'<w:jc w:val="center"/></w:pPr><w:r><w:rPr>'
            f'<w:rFonts w:ascii="Arial" w:cs="Arial" w:eastAsia="Arial" w:hAnsi="Arial"/>{negrita}'
            f'<w:color w:val="555555"/><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
            f'<w:t xml:space="preserve">{html.escape(linea, quote=False)}</w:t></w:r></w:p>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            + "".join(parrafos) + "</w:ftr>")


def agregar_pie(xml):
    """Enlaza el pie de página con los créditos en la sección del documento."""
    ref = f'<w:footerReference w:type="default" r:id="{RID_PIE}"/>'
    xml, n = re.subn(r"(<w:sectPr[^>]*>)", lambda mm: mm.group(1) + ref, xml)
    assert n == 1, n
    return xml


def escribir_docx(plantilla, destino, transformar):
    with zipfile.ZipFile(plantilla) as zin, zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            datos = zin.read(item.filename)
            if item.filename == "word/document.xml":
                datos = agregar_pie(transformar(datos.decode("utf-8"))).encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                datos = datos.decode("utf-8").replace(
                    "</Relationships>",
                    f'<Relationship Id="{RID_PIE}" Type="http://schemas.openxmlformats.org/officeDocument/'
                    f'2006/relationships/footer" Target="footer1.xml"/></Relationships>').encode("utf-8")
            elif item.filename == "[Content_Types].xml":
                datos = datos.decode("utf-8").replace(
                    "</Types>",
                    '<Override ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.'
                    'footer+xml" PartName="/word/footer1.xml"/></Types>').encode("utf-8")
            zout.writestr(item, datos)
        zout.writestr("word/footer1.xml", xml_pie_pagina())


def hoja_cmb(codigo, h, destino):
    def t(xml):
        xml = reemplazar_parrafo(xml, "Datos Observacionales CMB", f"Datos Observacionales CMB\nHoja {codigo}")
        # encabezado de la columna 4 con unidades correctas
        xml = re.sub(r"Fluctuación de temperatura \[</w:t>.*?\]</w:t>",
                     lambda mm: "Fluctuación de temperatura [μK²]</w:t>", xml, count=1, flags=re.S)
        xml = reemplazar_parrafo(xml, "Punto 1: Escala 220",
                                 f"¡Atención! El Astrónomo Observacional no puede mirar esta hoja. Debes dictar "
                                 f"los datos con voz clara (ej: \"Punto 1: Escala {h['x'][0]}, Intensidad {h['y'][0]}\").")
        filas = [[f"REG_{i+1:02d}", h["filtro"][i], h["x"][i], h["y"][i], h["calidad"][i]] for i in range(FILAS_CMB)]
        return rellenar_tabla(xml, filas)
    escribir_docx(PLANTILLAS / "Datos Observacionales CMB.docx", destino, t)


def columna_calidad_cef(xml):
    """En la tabla de Cefeidas reemplaza las columnas RA y DEC por una sola de «Calidad de señal»
    y quita la de Extinción."""
    t_ini = [m.start() for m in re.finditer("<w:tbl>", xml)][1]
    t_fin = xml.find("</w:tbl>", t_ini)
    tabla = xml[t_ini:t_fin]
    # anchos (twips) de: ID, Calidad, m, M, z  — suman el ancho original (8925)
    grilla = "".join(f'<w:gridCol w:w="{w}"/>' for w in [1925, 1925, 1600, 1600, 1875])
    tabla = re.sub(r"(<w:tblGrid>)(?:<w:gridCol [^>]*/>)+", lambda mm: mm.group(1) + grilla, tabla)
    for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tabla, re.S):
        tcs = re.findall(r"<w:tc>.*?</w:tc>", tr, re.S)
        assert len(tcs) == 7, len(tcs)
        nuevo = tr.replace(tcs[2], "", 1).replace(tcs[5], "", 1)  # fuera DEC y Extinción
        if "RA" in "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", tcs[1])):
            nuevo = nuevo.replace(tcs[1], set_texto_celda(tcs[1], "Calidad de señal"), 1)
        tabla = tabla.replace(tr, nuevo, 1)
    return xml[:t_ini] + tabla + xml[t_fin:]


def hoja_cef(codigo, h, destino):
    def t(xml):
        xml = reemplazar_parrafo(xml, "Datos Observacionales CMB", f"Datos Observacionales Cefeidas\nHoja {codigo}")
        xml = reemplazar_corrida(xml, "Satélite Planck (Órbita L2)", "Telescopio Espacial James Webb (Órbita L2)")
        xml = reemplazar_parrafo(xml, "Punto 1: Estrella 1",
                                 f"¡Atención! El Astrónomo Observacional no puede mirar esta hoja. Debes dictar "
                                 f"los datos con voz clara (ej: \"Cefeida 01: m = {coma(h['m'][0], 1)}, "
                                 f"M = {coma(h['M'][0], 1)}, z = {coma(h['z'][0], 4)}, "
                                 f"calidad {h['calidad'][0]}\").")
        xml = columna_calidad_cef(xml)
        xml = reemplazar_parrafo(xml, "Escala angular",
                                 "• ¿Qué indican la “magnitud aparente (m)” y la “magnitud absoluta (M)”?")
        xml = reemplazar_parrafo(xml, "menos confiables",
                                 "• ¿Qué estrellas no siguen la tendencia del gráfico? ¿Por qué podría pasar eso?")
        filas = [[f"Cefeida {i+1:02d}", h["calidad"][i], coma(h["m"][i], 1), coma(h["M"][i], 1),
                  coma(h["z"][i], 4)] for i in range(FILAS_CEF)]
        return rellenar_tabla(xml, filas)
    escribir_docx(PLANTILLAS / "Datos Observacionales Cefeidas.docx", destino, t)


# ----------------------------------------------------------------------------
def hoja_justa(generar, errores, real, H0):
    """Repite hasta obtener una hoja «justa» para la competencia:
    - sin outliers, el mejor ajuste es exactamente el valor real, y en el medidor del HTML
      se ve claramente (sin empate con los valores vecinos);
    - con todos los puntos, el ajuste queda a DESVIO_MIN–DESVIO_MAX del real;
    - si se olvida descartar CUALQUIERA de los outliers, el ajuste ya no da el real.
    """
    while True:
        h = generar(H0)
        n = len(h["calidad"])
        usar = np.ones(n, bool)
        usar[h["outliers"]] = False
        ajuste = lambda u: mejor_ajuste(*errores(h, u))
        aj_limpio = ajuste(usar)
        aj_todo = ajuste(np.ones(n, bool))
        if aj_limpio != real or not minimo_visible(*errores(h, usar), real) or not (DESVIO_MIN <= abs(aj_todo - real) <= DESVIO_MAX):
            continue
        olvido = {}
        for i in h["outliers"]:
            u = usar.copy()
            u[i] = True
            olvido[i] = ajuste(u)
        if all(v != real for v in olvido.values()):
            return h, aj_limpio, aj_todo, olvido


def html_control(datos):
    """Copia del HTML de los estudiantes que carga los datos de la hoja elegida y muestra la clave."""
    t = HTML_ESTUDIANTES.read_text(encoding="utf-8")
    cambios = [
        ('KEY = "h0-actividad-v2";', 'KEY = "h0-control-v2";'),
        ("<title>Midiendo la Expansión</title>", "<title>CONTROL · Midiendo la Expansión</title>"),
        ('<header class="top">',
         '<div class="panel no-print" style="border-color:var(--warn);background:#2a2410">'
         '<b style="color:var(--warn)">🔧 MODO CONTROL (profesor) — NO entregar a los estudiantes.</b> '
         'Elige el método y el código de la hoja: sus datos se cargan solos y verás la clave.</div>\n'
         '<header class="top">'),
        ('<section id="teamPanel" class="panel hidden">',
         '<section id="clavePanel" class="panel hidden" style="border-color:var(--warn)"></section>\n'
         '<section id="teamPanel" class="panel hidden">'),
        ('$("tHoja").addEventListener("change", function(e){ state.team.hoja=e.target.value; save(); });',
         '$("tHoja").addEventListener("change", function(e){ state.team.hoja=e.target.value; save(); cargarHoja(e.target.value); });'),
        ("/* ---------- Inicio ---------- */",
         "/* ---------- Modo control ---------- */\n"
         "var DATOS = " + json.dumps(datos, ensure_ascii=False) + ";\n"
         + r"""function cargarHoja(cod){
  var d=DATOS[cod], m=state.method; if (!d) { actualizarClave(); return; }
  state[m].rows = d.filas.map(function(f){ var o={use:true}; for (var k in f) o[k]=f[k]; return o; });
  state[m].H0 = m==="cmb" ? 40 : 60; state[m].final=null; lastErr[m]=null; save();
  fillTable(m); actualizarClave(); showStep(3);
}
function usarSegunClave(soloOutliers){
  var d=DATOS[state.team.hoja], m=state.method; if (!d) return;
  state[m].rows.forEach(function(r,i){ r.use = !(soloOutliers && d.outliers.indexOf(i)>=0); });
  state[m].final=null; save(); fillTable(m); renderAll();
}
function actualizarClave(){
  var p=$("clavePanel"), d=DATOS[state.team.hoja];
  p.classList.toggle("hidden", !state.method || !d);
  if (!d) return;
  var nom=function(i){ return d.metodo==="cmb" ? "P"+(i+1) : "Cefeida "+String(i+1).padStart(2,"0"); };
  var fx=function(v){ return fmt(v, d.metodo==="cmb"?0:1); };
  p.innerHTML='<h3 style="margin-top:0;color:var(--warn)">🔑 Clave de la hoja '+state.team.hoja+'</h3>'+
    '<p>Outliers: <b>'+d.outliers.map(nom).join(", ")+'</b></p>'+
    '<p>Descartando los '+d.outliers.length+' outliers → <b>'+fx(d.limpio)+'</b> km/s/Mpc (valor real del método: '+fmt(d.real,1)+')<br>'+
    'Sin descartar nada → '+fx(d.todo)+' km/s/Mpc<br>'+
    'Si olvidan descartar uno: '+d.outliers.map(function(i){ return nom(i)+' → '+fx(d.olvido[i]); }).join(" · ")+'</p>'+
    '<div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn small" id="cDesc">Descartar los outliers de la clave</button>'+
    '<button class="btn small ghost" id="cTodos">Volver a usar todos los puntos</button></div>';
  $("cDesc").onclick=function(){ usarSegunClave(true); };
  $("cTodos").onclick=function(){ usarSegunClave(false); };
}
var showMethodBase = showMethod;
showMethod = function(){ showMethodBase(); actualizarClave(); };

/* ---------- Inicio ---------- */"""),
    ]
    for viejo, nuevo in cambios:
        assert t.count(viejo) == 1, viejo
        t = t.replace(viejo, nuevo)
    HTML_CONTROL.write_text(t, encoding="utf-8")


def main():
    (AQUI / "CMB").mkdir(exist_ok=True)
    (AQUI / "Cefeidas").mkdir(exist_ok=True)

    real_cmb = round(H0_CMB_REAL)          # el deslizador del CMB avanza de 1 en 1
    real_cef = round(H0_CEF_REAL * 2) / 2  # el de Cefeidas, de 0,5 en 0,5

    clave = []
    filas_csv = []
    datos_html = {}

    for n in range(1, N_CMB + 1):
        codigo = f"CMB-{n}"
        h, aj_limpio, aj_todo, olvido = hoja_justa(generar_cmb, errores_cmb, real_cmb, H0_CMB_REAL)
        aj_limpio, aj_todo, olvido = int(aj_limpio), int(aj_todo), {i: int(v) for i, v in olvido.items()}
        hoja_cmb(codigo, h, AQUI / "CMB" / f"Hoja {codigo}.docx")
        nombre = lambda i: f"REG_{i+1:02d}"
        clave.append(dict(codigo=codigo, aj_limpio=aj_limpio, aj_todo=aj_todo,
                          outliers=[nombre(i) for i in h["outliers"]],
                          olvido=[f"{nombre(i)} → {v}" for i, v in olvido.items()]))
        datos_html[codigo] = dict(metodo="cmb", real=H0_CMB_REAL, limpio=aj_limpio, todo=aj_todo,
                                  outliers=h["outliers"], olvido={int(i): v for i, v in olvido.items()},
                                  filas=[dict(x=str(h["x"][i]), y=str(h["y"][i]), q=h["calidad"][i]) for i in range(FILAS_CMB)])
        for i in range(FILAS_CMB):
            filas_csv.append([codigo, f"REG_{i+1:02d}", h["filtro"][i], h["x"][i], h["y"][i], h["calidad"][i],
                              "", "", "", "si" if i in h["outliers"] else "no"])

    for n in range(1, N_CEF + 1):
        codigo = f"CEF-{n}"
        h, aj_limpio, aj_todo, olvido = hoja_justa(generar_cef, errores_cef, real_cef, H0_CEF_REAL)
        hoja_cef(codigo, h, AQUI / "Cefeidas" / f"Hoja {codigo}.docx")
        nombre = lambda i: f"Cefeida {i+1:02d}"
        clave.append(dict(codigo=codigo, aj_limpio=coma(aj_limpio, 1), aj_todo=coma(aj_todo, 1),
                          outliers=[nombre(i) for i in h["outliers"]],
                          olvido=[f"{nombre(i)} → {coma(v, 1)}" for i, v in olvido.items()]))
        datos_html[codigo] = dict(metodo="cef", real=H0_CEF_REAL, limpio=aj_limpio, todo=aj_todo,
                                  outliers=h["outliers"], olvido={int(i): v for i, v in olvido.items()},
                                  filas=[dict(m=coma(h["m"][i], 1), M=coma(h["M"][i], 1),
                                              z=coma(h["z"][i], 4), q=h["calidad"][i]) for i in range(FILAS_CEF)])
        for i in range(FILAS_CEF):
            filas_csv.append([codigo, f"Cefeida {i+1:02d}", "", "", "", h["calidad"][i], coma(h["m"][i], 1), coma(h["M"][i], 1),
                              coma(h["z"][i], 4), "si" if i in h["outliers"] else "no"])

    with open(AQUI / "datos_todas_las_hojas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["hoja", "id", "filtro_GHz", "escala_angular", "fluctuacion_uK2", "calidad",
                    "m", "M", "z", "outlier"])
        w.writerows(filas_csv)

    html_control(datos_html)

    with open(AQUI / "CLAVE_PROFESOR.md", "w", encoding="utf-8") as f:
        f.write("# Clave del profesor — NO IMPRIMIR NI ENTREGAR A LOS ESTUDIANTES\n\n")
        f.write(f"Valores de referencia: CMB (Planck 2018) H0 = {H0_CMB_REAL} km/s/Mpc · "
                f"Cefeidas (SH0ES 2022) H0 = {H0_CEF_REAL} km/s/Mpc.\n\n")
        f.write("**Todas las hojas se generaron con el valor real de su método.** Ninguna hoja tiene ventaja: "
                f"todas las de CMB tienen {FILAS_CMB} puntos y {OUTLIERS_CMB} outliers, y todas las de Cefeidas "
                f"{FILAS_CEF} estrellas y {OUTLIERS_CEF} outliers. Solo descartándolos TODOS se llega al valor real "
                f"(CMB: {real_cmb}; Cefeidas: {coma(real_cef, 1)}, porque el deslizador del CMB avanza de 1 en 1 y el de "
                "Cefeidas de 0,5 en 0,5). Si un equipo olvida aunque sea un outlier, su resultado ya se aleja. "
                "Así gana el criterio para elegir y ajustar los datos, no la suerte de la hoja.\n\n")
        f.write("«Sin descartar nada» es el H0 al que lleva el medidor de error si se usan todos los puntos. "
                "«Si olvidan uno» indica qué resultado obtiene un equipo que descarta los demás outliers pero no ese. "
                "Estos dos valores son aproximados (±1 paso del deslizador), porque con outliers el mínimo es más plano.\n\n")
        f.write("| Hoja | Descartando los outliers | Sin descartar nada | Outliers | Si olvidan uno |\n")
        f.write("|---|---|---|---|---|\n")
        for c in clave:
            f.write(f"| {c['codigo']} | **{c['aj_limpio']}** | {c['aj_todo']} | {', '.join(c['outliers'])} | "
                    f"{' · '.join(c['olvido'])} |\n")
        f.write("\n## Para revisar rápido\n\n"
                "Abre «CONTROL profesor - Actividad H0.html» (en esta carpeta): al elegir el código de una hoja, "
                "sus datos se cargan solos y aparece esta clave con un botón para descartar los outliers.\n\n"
                "## Si hay empate\n\n"
                "Como todas las hojas pueden llegar al valor real, es probable que varios equipos empaten. "
                "Para desempatar, pidan que expliquen qué puntos descartaron y por qué (preguntas del paso 4): "
                "gana quien lo justifique mejor.\n\n"
                "## Para el cierre: la «tensión de Hubble»\n\n"
                f"Los equipos del CMB llegarán a ≈ {H0_CMB_REAL} y los de Cefeidas a ≈ {H0_CEF_REAL} km/s/Mpc. "
                "Esa diferencia es real y es uno de los grandes misterios actuales de la cosmología: dos métodos muy "
                "cuidadosos no se ponen de acuerdo. Buena pregunta para el curso: ¿error de medición o física nueva?\n")
    for c in clave:
        print(c)


if __name__ == "__main__":
    main()
