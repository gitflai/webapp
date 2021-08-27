import streamlit as st 
import pandas as pd
from pycaret.classification import load_model, predict_model



paginas = ['Home', 'Precificação de Seguro', 'Detector de Fumante', 'Sobre']
pagina = st.sidebar.radio('Selecione uma página', paginas)

#n = st.sidebar.slider('Entre com um número', 10, 20, 12, 2)
#st.text('O número selecionado foi {}'.format(n))
#st.text('O quadrado do numero é {}'.format(n**2))


if pagina == 'Home':
	st.title('Deploy de Modelos - Instituição de Seguro de Saúde')
	st.subheader('Powered by Streamlit')
	st.markdown('---')
	st.markdown('Web App para utilização prática de dois modelos. Selecione o modelo que deseja utilizar agora no menu ao lado.')





if pagina == 'Precificação de Seguro':
	st.markdown('# Precificação de Seguro')
	st.markdown('---')
	idade = st.slider('Entre com a idade:', 18, 65, 30)
	sexo = st.selectbox('Entre com o sexo:', ['male', 'female'])
	imc = st.number_input('IMC:', 15, 55, 25)
	criancas = st.selectbox("Dependentes", [0, 1, 2, 3, 4, 5])
	fumante = st.selectbox("É fumante?", ['yes', 'no'])
	regiao = st.selectbox("Região em que mora", 
								  ['southeast', 'southwest', 'northeast', 'northwest'])

	st.markdown('---')

	dados_dicio = {'age': [idade], 'sex': [sexo], 'bmi': [imc], 
			'children': [criancas], 'region': [regiao], 'smoker': [fumante]}

	dados = pd.DataFrame(dados_dicio)	
 
	if st.button('CLIQUE AQUI PARA EXECUTAR O MODELO'):
		modelo = load_model('meu-melhor-modelo-para-charges')
		saida = predict_model(estimator = modelo, data = dados) 
		pred = float(saida['Label'].round(2)) 
		s1 = 'Custo Estimado do Seguro: ${:.2f}'.format(pred) 
		st.markdown('## Resultados do modelo') 
		st.markdown('## **' + s1 + '**')  



if pagina == 'Detector de Fumante':
	st.markdown('# 🚧 Under construction... ')



if pagina == 'Sobre':
	st.markdown('### Aplicativo de exemplificação de deploy de modelos com streamlit para o pessoal da Turma 5 do DDS da FLAI')