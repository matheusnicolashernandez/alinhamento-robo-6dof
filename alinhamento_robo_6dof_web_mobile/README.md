# Alinhamento Robô 6 DOF + 4 Lasers — Web

Esta pasta contém uma versão web do simulador que pode ser aberta em
celular, tablet ou computador através do navegador.

## Arquivos

- `app.py` — aplicação Streamlit.
- `normal.urdf` — URDF usado como base da cadeia cinemática.
- `requirements.txt` — dependências.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar sem deixar o PC ligado

A forma mais simples é usar o Streamlit Community Cloud:

1. Crie um repositório no GitHub.
2. Envie para o repositório:
   - `app.py`
   - `normal.urdf`
   - `requirements.txt`
3. Entre no Streamlit Community Cloud.
4. Conecte sua conta GitHub.
5. Crie um novo app apontando para `app.py`.

Depois da publicação, o Streamlit fornece uma URL para abrir no celular.

O `normal.urdf` deve permanecer no mesmo diretório de `app.py`.

## Observação sobre o modelo

A versão web mantém a lógica usada na simulação desktop:
- 6 juntas;
- J1 diretamente na base;
- J6 como ponto de acoplamento da ferramenta;
- retângulo centralizado na J6;
- rotação fixa de 90° do retângulo no próprio plano;
- Z do end-effector como normal/perpendicular dos lasers;
- tubo configurável em X, Y, Z, diâmetro e comprimento;
- base configurável em X, Y e Z;
- controlador pela Jacobiana dos quatro lasers;
- gráfico A/B/C/D.
