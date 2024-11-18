from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Titre principal de l'application
st.markdown("<h1 class='title'>🏅 Analyse des Prix Nobel</h1>", unsafe_allow_html=True)

st.sidebar.image(
    "icon.png",
    caption="Mini Projet d'Analyse de Donnée",
    width=150
)

st.sidebar.title("🌟 Navigation 🌟")
page = st.sidebar.radio(
    "Aller à :",
    [
        "🏠 Page d'introduction",
        "📊 Tables",
        "📈 Visualisations"
    ]
)


st.sidebar.markdown(
    """
    ---
    🛠️ **Projet sur [GitHub](https://github.com/votre-repo)** 🛠️
    """
)
df = pd.read_excel("nobel-prize-laureates.xlsx")
prix_nobel = pd.DataFrame(df,columns = ['Category','country'])




encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(prix_nobel)

# Calcul du tableau de Burt
burt_matrix = (encoded_data.T @ encoded_data).toarray()  # Convertir en array pour faciliter l'affichage
burt_df = pd.DataFrame(burt_matrix, columns=encoder.get_feature_names_out(), index=encoder.get_feature_names_out())


def mesure_ressemblance_vectorisee(tab_codage):
    # Convertir la matrice CSR en tableau NumPy si nécessaire
    if not isinstance(tab_codage, np.ndarray):
        tab_codage = tab_codage.toarray()

    # Comparaison des lignes par paires
    comparaison = np.equal(tab_codage[:, None, :], tab_codage[None, :, :])

    # Calcul de la ressemblance en divisant la somme des égalités par la taille de chaque ligne
    ressemblance = comparaison.sum(axis=2) / tab_codage.shape[1]

    # Convertir en DataFrame (assurez-vous d'avoir un index clair pour vos lignes)
    return pd.DataFrame(ressemblance)


# Exécuter la fonction sur votre matrice CSR ou DataFrame
tab_distanceR = mesure_ressemblance_vectorisee(encoded_data)


def mesure_dissemblance_vectorisee(b_df):
    # Convertir la matrice CSR en tableau NumPy si nécessaire
    if not isinstance(b_df, np.ndarray):
        b_df = b_df.toarray()

    # Comparaison des lignes par paires
    comparaison = np.not_equal(b_df[:, None, :], b_df[None, :, :])

    # Calcul de la dissemblance en divisant la somme des différences par la taille de chaque ligne
    dissemblance = comparaison.sum(axis=2) / b_df.shape[1]

    # Convertir en DataFrame (assurez-vous que votre index est défini si vous avez un DataFrame initialement)
    return pd.DataFrame(dissemblance)



tab_distanceD = mesure_dissemblance_vectorisee(encoded_data)


table_contingence = burt_df.loc[['Category_Chemistry', 'Category_Economics', 'Category_Literature', 'Category_Medicine', 'Category_Peace',
       'Category_Physics'], ['country_Australia', 'country_Austria', 'country_Belgium', 'country_Canada', 'country_China', 'country_Denmark',
       'country_France', 'country_Germany', 'country_India', 'country_Ireland', 'country_Italy', 'country_Japan',
       'country_Norway', 'country_Poland', 'country_Russia', 'country_South Africa', 'country_Spain', 'country_Sweden',
       'country_Switzerland', 'country_USA', 'country_United Kingdom', 'country_other world',
       'country_the Netherlands']]



def tab_freq(cont_tab):
    K = cont_tab.values.sum()
    rows = cont_tab.shape[0]
    columns = cont_tab.shape[1]

    tab_freq = pd.DataFrame(np.zeros((rows, columns)), index=cont_tab.index, columns=cont_tab.columns)

    for i in range(rows):
        for j in range(columns):
            tab_freq.iloc[i, j] = cont_tab.iloc[i, j] / K

    return tab_freq

table_frequence = tab_freq(table_contingence)


def tab_profile_ligne(tab_freq):
    tab_profile_ligne = tab_freq
    for i in range(tab_freq.shape[0]):
        f_i_point = tab_freq.iloc[i].sum()
        for j in range(tab_freq.shape[1]):
            tab_profile_ligne.iloc[i,j] = tab_freq.iloc[i,j]/f_i_point
    return tab_profile_ligne

tab_profile_ligne_ = tab_profile_ligne(table_frequence)





def tab_profile_colonne(tab_freq):
    tab_prof_colonne = tab_freq.copy()
    for i in range(tab_freq.shape[0]):
        for j in range(tab_freq.shape[1]):
            f_point_j = tab_freq.iloc[:,j].sum()
            tab_prof_colonne.iloc[i,j] = tab_freq.iloc[i,j]/f_point_j
    return tab_prof_colonne

tab_profile_colonne_ = tab_profile_colonne(table_frequence)



def Nuage_I(tab_p_l, tab_f):
    NI = []

    for i in range(tab_f.shape[0]):
        FiJ = []
        FiJpoids = []
        f_i_point = tab_f.iloc[i].sum()

        for j in range(tab_f.shape[1]):
            x = tab_p_l.iloc[i, j]
            FiJ.append(x)

        FiJpoids.append(FiJ)
        FiJpoids.append(f_i_point)
        NI.append(FiJpoids)

    return NI


N_I = Nuage_I(tab_profile_ligne_, table_frequence)



def Nuage_J(tab_p_c, tab_f):
    NJ = []

    for j in range(tab_f.shape[1]):
        FjI = []
        FjIpoids = []
        f_point_j = tab_f.iloc[:, j].sum()

        for i in range(tab_f.shape[0]):
            x = tab_p_c.iloc[i, j]
            FjI.append(x)

        FjIpoids.append(FjI)
        FjIpoids.append(f_point_j)
        NJ.append(FjIpoids)

    return NJ


N_J = Nuage_J(tab_profile_colonne_, table_frequence)


# Fonction pour convertir le DataFrame en fichier Excel en mémoire
def convert_df_to_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Données')
    processed_data = output.getvalue()
    return processed_data




def gNI_gNJ (NI,NJ):
    gi = []
    gj = []
    for i in range(len(NJ)):
        gi.append(NJ[i][1])
    for i in range(len(NI)):
        gj.append(NI[i][1])
    return gi , gj



gi1,gj1 = gNI_gNJ(N_I,N_J)


def format_nested_list(nested_list):
    """Formate les nombres dans une liste (y compris les listes imbriquées) à deux chiffres après la virgule."""
    if isinstance(nested_list, (int, float)):  # Si c'est un nombre
        return f"{nested_list:.2f}"
    elif isinstance(nested_list, list):  # Si c'est une liste, parcourir récursivement
        return [format_nested_list(item) for item in nested_list]
    else:
        return nested_list  # Laisser les autres types intacts



tab_codage = pd.get_dummies(prix_nobel,columns=[
    'Category',
    'country'
])  # tableau de codage avec (True/False)

tab_codage = tab_codage.astype(int)




def afficher_histogramme(table, title):
    # Conversion en format long
    table.index.name = "Lignes"
    table_long = table.reset_index().melt(id_vars="Lignes",
                                             var_name="Colonnes",
                                             value_name="Fréquences")

    # Création du graphique à barres
    fig = px.bar(
        table_long,
        x="Colonnes",
        y="Fréquences",
        color=table.index.name,
        labels={"Colonnes": "Colonnes", table.index.name: "Lignes", "Fréquences": "Fréquences"},
        title=title,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Ajustement des dimensions
    #fig.update_layout(width=500, height=300)

    # Affichage du graphique
    st.plotly_chart(fig)














# Afficher le contenu en fonction de la sélection
if page == "🏠 Page d'introduction":
    st.markdown("<h3 class='title'>Introduction du Projet</h3>", unsafe_allow_html=True)
    st.write("Bienvenue sur l'analyse des données des Prix Nobel !")
    st.write("""
        Ce projet vise à analyser et visualiser les données des lauréats des prix Nobel.
        Vous trouverez ici :
        - Les tableaux de données des prix Nobel
        - Des visualisations interactives
    """)




elif page == "📊 Tables":
    st.markdown("<h3 class='title'>Visualisation des Tables</h3>", unsafe_allow_html=True)
    st.write("Ici, vous pouvez consulter les tables de données analysées.")
    # Ajoutez du code pour afficher les tables

    st.markdown("<h3 class='title'>DataFrame :</h3>", unsafe_allow_html=True)
    st.dataframe(prix_nobel.head(50))

    excel_data = convert_df_to_excel(prix_nobel)# Convertir le DataFrame en Excel


    st.download_button(     # Bouton de téléchargement pour le fichier Excel
        label="Télécharger les données en Excel",
        data=excel_data,
        file_name="datframeExcel.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_button_1"
    )


    #table_html = prix_nobel.to_html(index=False)
    #st.markdown(f'<div class="center-table">{table_html}</div>', unsafe_allow_html=True)


    st.markdown("<h3 class='title'>Tableau de Codage disjontif :</h3>", unsafe_allow_html=True)
    #st.markdown('<div class="center-table-container">', unsafe_allow_html=True)
    #st.table(tab_codage)
    st.dataframe(tab_codage.head(50))
    excel_data = convert_df_to_excel(tab_codage)

    # Bouton de téléchargement pour le fichier Excel
    st.download_button(
        label="Télécharger les données en Excel",
        data=excel_data,
        file_name="distanceR_matrix.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_button_2"
    )
    #st.table(tab_distanceR)
   # st.dataframe(tab_codage.iloc[:50, :50])
    #st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h4 class='title'>Tableau de Distance - en utilisant la mesure de ressemblance -  :</h4>", unsafe_allow_html=True)
    #st.markdown('<div class="center-table-container">', unsafe_allow_html=True)
    st.dataframe(tab_distanceR.iloc[:50, :50])

    excel_data = convert_df_to_excel(tab_distanceR)


    st.download_button(
        label="Télécharger les données en Excel",
        data=excel_data,
        file_name="distanceR_matrix.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_button_3"
    )
    #st.table(tab_distanceR)

    #st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h4 class='title'>Tableau de Distance - en utilisant la mesure de dissemblance -  :</h4>", unsafe_allow_html=True)
    #st.table(tab_distanceD)
    st.dataframe(tab_distanceD.iloc[:50, :50])

    excel_data = convert_df_to_excel(tab_distanceD)

    # Bouton de téléchargement pour le fichier Excel
    st.download_button(
        label="Télécharger les données en Excel",
        data=excel_data,
        file_name="distanceD_matrix.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_button_4"
    )


    st.markdown("<h4 class='title'>Tableau de Burt :</h4>", unsafe_allow_html=True)
    st.table(burt_df)


    st.markdown("<h4 class='title'>Tableau de Contingence :</h4>", unsafe_allow_html=True)

    table_html = table_contingence.to_html(index=False)
    st.markdown(f'<div class="center-table">{table_html}</div>', unsafe_allow_html=True)




    st.markdown("<h4 class='title'>Tableau Profiles Ligne :</h4>", unsafe_allow_html=True)
    table_html = tab_profile_ligne_.to_html(index=False)
    st.markdown(f'<div class="center-table">{table_html}</div>', unsafe_allow_html=True)
    st.write("Les nuages :")

    formatted_N_I = format_nested_list(N_I)
    st.write("- **Colonnes N(I)**: ", formatted_N_I)
    st.write("Centre de Gravité :")
    formatted_list = [f"{x:.2f}" for x in gi1]
    st.write(f"- **Lignes**: {formatted_list}")





    st.markdown("<h4 class='title'>Tableaux Profiles Colonnes :</h4>", unsafe_allow_html=True)
    st.markdown("<h5 class='title'>Tableau profile Colonne rep1_2 :</h5>", unsafe_allow_html=True)
    table_html = tab_profile_colonne_.to_html(index=False)
    st.markdown(f'<div class="center-table">{table_html}</div>', unsafe_allow_html=True)
    st.write("Les nuages :")
    formatted_N_J = format_nested_list(N_J)
    st.write("- **Colonnes N(J)**: ", formatted_N_J)  # Formater chaque élément

    st.write("Centre de Gravité :")
    formatted_list = [f"{x:.2f}" for x in gj1]
    st.write(f"- **Colonnes**: {formatted_list}")





















elif page == "📈 Visualisations":
    st.markdown("<h3 class='title'>Visualisation des Données</h3>", unsafe_allow_html=True)
    st.write("Ici, vous pouvez voir des visualisations des données.")
    st.markdown("<h4 class='title'>Representation graphique de la distribution des valeurs de categories et pays :</h4>", unsafe_allow_html=True)

    col1_counts = prix_nobel['Category'].value_counts().reset_index()
    col1_counts.columns = ["Catégorie", "Fréquence"]

    fig1 = px.pie(
        col1_counts,
        names="Catégorie",
        values="Fréquence",
        title="Distribution des valeurs de Category",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig1.update_traces(textinfo="percent")

    # Créer un graphique en camembert pour Colonne2
    col2_counts = prix_nobel['country'].value_counts().reset_index()
    col2_counts.columns = ["Catégorie", "Fréquence"]

    fig2 = px.pie(
        col2_counts,
        names="Catégorie",
        values="Fréquence",
        title="Distribution des valeurs des Contry",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig2.update_traces(textinfo="percent")
    fig2.update_layout(width=800, height=600)

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)


    st.markdown("<h4 class='title'>Representation graphique du tableau de Burt :</h4>", unsafe_allow_html=True)

    st.markdown("<h5 class='title'>Representation sous forme de pie:</h5>", unsafe_allow_html=True)

    tab_burt_aggregated = burt_df.sum(axis=1)  # Somme les fréquences par lignes pour simplifier

    # Créez un DataFrame pour représenter ces données agrégées
    tab_burt_pie_data = tab_burt_aggregated.reset_index()
    tab_burt_pie_data.columns = ["Catégorie", "Fréquence Totale"]

    # Création du graphique en secteurs (camembert)
    fig = px.pie(
        tab_burt_pie_data,
        names="Catégorie",
        values="Fréquence Totale",
        title="Distribution des Fréquences Agrégées du Tableau de Burt",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Affichage des pourcentages sur chaque secteur
    fig.update_traces(textinfo="percent+label", textposition='inside')

    # Ajustez la taille du graphique
    fig.update_layout(width=800, height=600)

    # Affichage du graphique
    st.plotly_chart(fig)



    st.markdown("<h4 class='title'>Representation graphique des tableaux de Contingence :</h4>", unsafe_allow_html=True)
    st.markdown("<h5 class='title'> Table de contingence  :</h5>", unsafe_allow_html=True)


    # Création du graphique en secteurs (camembert)


    table_contingence_aggregated = table_contingence.sum(axis=1)  # Somme les fréquences par lignes pour simplifier

    # Créez un DataFrame pour représenter ces données agrégées
    table_contingence_pie_data = table_contingence_aggregated.reset_index()
    table_contingence_pie_data.columns = ["Catégorie", "Fréquence Totale"]

    fig = px.pie(
        table_contingence_pie_data,
        names="Catégorie",
        values="Fréquence Totale",
        title="Distribution des Fréquences Agrégées du Tableau de Contingence",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Affichage des pourcentages sur chaque secteur
    fig.update_traces(textinfo="percent+label", textposition='inside')

    # Ajustez la taille du graphique
    fig.update_layout(width=800, height=600)

    # Affichage du graphique
    st.plotly_chart(fig)

    afficher_histogramme(table_contingence, "Table de Contingence:")


    # Exemple des nuages de points I et J
    nuages_I = [[[0.6666666666666666, 0.3333333333333333], 1.0], [[0.5, 0.5], 1.0]]
    nuages_J = [[[1.5, 1.0], 1.0], [[1.2, 1.2], 1.0]]

    # Centre de gravité
    centre_gravite = [1.1666666666666665, 0.8333333333333333]

    # Extraction des coordonnées des nuages I
    points_I_x = [point[0][0] for point in nuages_I]
    points_I_y = [point[0][1] for point in nuages_I]

    # Extraction des coordonnées des nuages J
    points_J_x = [point[0][0] for point in nuages_J]
    points_J_y = [point[0][1] for point in nuages_J]

    # Création du graphique
    plt.figure(figsize=(8, 6))
    plt.scatter(points_I_x, points_I_y, color='blue', label="Nuage I", s=100)
    plt.scatter(points_J_x, points_J_y, color='red', label="Nuage J", s=100)
    plt.scatter(centre_gravite[0], centre_gravite[1], color='green', label="Centre de Gravité", marker='X', s=200)

    # Ajout des labels et du titre
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title("Représentation des Nuages de Points et Centre de Gravité")
    plt.legend()
    plt.grid()

    # Affichage
    plt.show()

    # Ajoutez du code pour afficher les visualisations

# Fin du script
