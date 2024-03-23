from main_model import MainModel
from typing import Dict,List
import pandas as pd
import os


class ModelEvaluator:

    eval_file = "../evaluation/evaluation.csv"

    @classmethod
    def new_query(cls, input_query, expected_output, expected_retrieved_k, verbose: bool = False):
        single_dict = {'method' : 'single',
            'input': input_query, 'expected_output': expected_output, 'expected_documents': expected_retrieved_k
        }

        double_dict = {'method' : 'double',
            'input': input_query, 'expected_output': expected_output, 'expected_documents': expected_retrieved_k
        }

        single_out, single_docs = MainModel.query(input_query, return_found_docs=True)

        if verbose:
            print("Single step query completed.")
            print()

        double_out, double_docs = MainModel.double_step_query(input_query, return_found_docs=True)

        if verbose:
            print("Double step query completed.")
            print()

        single_dict['actual_output'] = single_out
        double_dict['actual_output'] = double_out

        single_dict['actual_documents'] = single_docs
        double_dict['actual_documents'] = double_docs

        if not os.path.isfile(cls.eval_file):
            df = pd.DataFrame(data = single_dict,index = [0])
            df.loc[len(df)] = double_dict
        else:
            df = pd.read_csv(cls.eval_file)

            df.loc[len(df)] = single_dict
            df.loc[len(df)] = double_dict

        df.to_csv(cls.eval_file, index = False)

    @classmethod
    def evaluate(cls, metrics: List[str]):
        pass  # TODO


if __name__ == '__main__':
#     q01 = """1-	Per quale motivo il parere di compatibilita’
#     idrulica ha dato esito negativo?"""
#
#     a01 = """le spalle ricadono in area Hi4. Tale condizione risulta in contrasto con la
#      previsione delle Norme Tecniche delle costruzioni 2018 """
#
#     q02 = """è presente tra la documentazione la verifica del rispetto dei franchi idraulici?"""
#
#     a02 = "no"
#
#     q03 = """che modifica viene apportata con la variante giliacquas al PUC in particolare all’articolo 11? """
#
#     a03 = """Indice di edificabilita’ territoriale 0,4mc/m2"""
#
#     q04 = """che cosa e’ previsto nella variante al PUC per la sottozona H2.1?"""
#
#     a04 = """Nella sottozona sono ammessi gli interventi volti al miglioramento della fruizione e funzionali alla riduzione degli impatti su habitat e specie.
# L’indice di edificabilità fondiario è stabilito in 0,001 mc/mq.
# Tutte le volumetrie devono essere previste con soluzioni facilmente amovibili e capaci di non incidere sul sottosuolo e del tipo NBS (Natural Base Solution).
# Per la realizzazione di viabilità e percorsi devono impiegarsi tecniche e pavimentazioni Sustainable drainage system (SuDS), tali da conservare la capacità drenante dei suoli.
# Gli interventi di scavo sono attuati in coerenza con il Piano di Assetto Idrogeologico e nelle aree di interesse archeologico non superano la profondità di 50 cm.
# E’ vietato l’impiego di specie vegetali invasive e non appartenenti agli habitat di riferimento così come elencati nei Piani di Gestione di Rete Natura 2000.
# """
#
#     q05 = """Quali sono le caratteristiche geomorfologiche dell’area?"""
#
#     a05 = """costituisce essenzialmente un grande bacino retrocostiero, separato dal mare e dalle dinamiche di spiaggia
#     da uno stretto cordone litoraneo, lungo quasi dieci chilometri, la cui genesi è riferibile alla emersione di barre
#      sabbiose avvenuta in relazione alle ultime fasi di oscillazione del livello marino di base, successive al culmine
#      trasgressivo versiliano """
#
#     q06 = """Quali sono i vincoli gravanti sui terreni oggetto del progetto?"""
#
#     a06 = None
#
#     q07 = """Che cos’è il franco idraulico?"""
#
#     a07 = """rappresenta l’altezza verticale aggiuntiva da considerare in fase di dimensionamento della sezione di
#     deflusso rispetto al livello idrico corrispondente alla portata di progetto"""
#
#     q08 = """A che quota devono essere collocati gli intradossi degli attraversamenti di progetto?"""
#
#     a08 = """0 ad 1 m"""
#
#     q_list = [q01,q02,q03,q04,q05,q06,q07,q08]
#     a_list = [a01,a02,a03,a04,a05,a06,a07,a08]
#
#     for i in range(len(q_list)):
#         ModelEvaluator().new_query(input_query=q_list[i], expected_output=a_list[i], expected_retrieved_k=None)

