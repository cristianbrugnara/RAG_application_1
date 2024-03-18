from main_model import MainModel
from typing import Dict,List


class ModelEvaluator:
    __single_query_performance: List[Dict] = []
    __double_query_performance: List[Dict] = []

    @classmethod
    def get_single_query_performance(cls):
        return cls.__single_query_performance

    @classmethod
    def get_double_query_performance(cls):
        return cls.__double_query_performance

    @classmethod
    def reset(cls):
        cls.__single_query_performance.clear()
        cls.__double_query_performance.clear()

    @classmethod
    def new_query(cls, input_query, expected_output, expected_retrieved_k, verbose: bool = False):
        single_dict = {
            'input': input_query, 'expected_output': expected_output, 'expected_documents': expected_retrieved_k
        }

        double_dict = {
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

        cls.__single_query_performance.append(single_dict)
        cls.__double_query_performance.append(double_dict)

    @classmethod
    def evaluate(cls, metrics: List[str]):
        pass  # TODO


if __name__ == '__main__':

    q03 = """Per quale motivo la Direzione Generale dell 'Agenzia Regionale del Distretto Idrografico della Sardegna ha 
    espresso parere di non approvazione dello Studio di compatibilità idraulica relativo al progetto "PNRR - M5C2 - 
    INV. 2.2" comunicato con prot. n. 6133 del 26/4/2023?"""

    a03 = """Il parere di non approvazione è stato espresso perché le opere previste ricadevano in aree a pericolosità 
    idraulica di livello Hi1 e Hi4, e le spalle dei ponti di nuova realizzazione ricadevano in area Hi4, in contrasto 
    con le Norme Tecniche delle costruzioni 2018.
    """

    ModelEvaluator.new_query(input_query=q03,expected_output=a03,expected_retrieved_k=None, verbose=True)

    print()

    print("Single query history:")
    print(ModelEvaluator.get_single_query_performance())
    print()
    print("Double query history:")
    print(ModelEvaluator.get_double_query_performance())

    # ModelEvaluator.reset()
