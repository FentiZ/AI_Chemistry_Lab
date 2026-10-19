import csv
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.error')

FOLDER = Path(__file__).resolve().parent
CSV_FILE = FOLDER / 'chemistry_results.csv'
REPORT_FILE = FOLDER / 'report.txt'
MODEL_NAME = 'khanfs/ChemSolubilityBERTa'
PROPERTY = 'Water solubility (log10 mol/L)'
COLUMNS = ['molecule_name', 'formula', 'smiles', 'predicted_property', 'predicted_value']


EXAMPLES = [
    ('Этанол', 'C2H6O', 'CCO'),
    ('Уксусная кислота', 'C2H4O2', 'CC(=O)O'),
    ('Бензол', 'C6H6', 'C1=CC=CC=C1'),
    ('Метанол', 'CH4O', 'CO'),
    ('Ацетон', 'C3H6O', 'CC(=O)C'),
    ('Толуол', 'C7H8', 'Cc1ccccc1'),
    ('Фенол', 'C6H6O', 'Oc1ccccc1'),
    ('Этиленгликоль', 'C2H6O2', 'OCCO'),
    ('Глицерин', 'C3H8O3', 'OCC(O)CO'),
    ('Гексан', 'C6H14', 'CCCCCC'),
]

model = None
tokenizer = None


def get_model():
    global model, tokenizer
    if model is None:
        print('\nЗагрузка AI-модели с Hugging Face (первый запуск требует интернет)...')
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            model.eval()
        except Exception as error:
            raise RuntimeError('Не удалось загрузить модель. Проверьте интернет и библиотеки.\n' + str(error))
    return model, tokenizer


def check_molecule(formula, smiles):
    if not formula or not smiles:
        raise ValueError('Формула и SMILES не могут быть пустыми.')
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Неверный SMILES!')
    real_formula = rdMolDescriptors.CalcMolFormula(molecule)

    subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    if formula.translate(subscripts).strip() != real_formula:
        raise ValueError('Формула не совпадает с SMILES. Правильная формула: ' + real_formula)


def predict(smiles):
    import torch
    ai_model, ai_tokenizer = get_model()
    inputs = ai_tokenizer(smiles, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = ai_model(**inputs)
    return round(float(output.logits[0][0].item()), 4)


def read_results():
    if not CSV_FILE.exists():
        return []
    with open(CSV_FILE, 'r', encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def save_results(rows):
    with open(CSV_FILE, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def add_result(rows, name, formula, smiles):
    check_molecule(formula, smiles)
    value = predict(smiles)
    new_row = {
        'molecule_name': name,
        'formula': formula,
        'smiles': smiles,
        'predicted_property': PROPERTY,
        'predicted_value': value,
    }
    for index, row in enumerate(rows):
        if row['molecule_name'].casefold() == name.casefold():
            rows[index] = new_row
            break
    else:
        rows.append(new_row)
    save_results(rows)
    print(f'{name}: растворимость = {value:.4f} log10(моль/л)')


def ready_rows(rows):
    return [row for row in rows if str(row.get('predicted_value') or '').strip()]


def show_rows(rows):
    if not rows:
        print('Нет результатов. Выберите анализ 10 веществ (пункт 2).')
        return
    print('\nНазвание                 Формула       Растворимость (log10 моль/л)')
    print('-' * 69)
    for row in rows:
        print(f"{row['molecule_name'][:23]:<24} {row['formula']:<13} {float(row['predicted_value']):>10.4f}")


def write_report(rows):
    ready = ready_rows(rows)
    report = [
        'ПРАКТИЧНА РОБОТА: AI CHEMISTRY LAB',
        '',
        '1. Обрана модель',
        'Назва: khanfs/ChemSolubilityBERTa',
        'Посилання: https://huggingface.co/khanfs/ChemSolubilityBERTa',
        'Це модель ChemBERTa, додатково навчена на наборі ESOL',
        '(1128 молекул) для прогнозування розчинності речовин у воді.',
        'Її обрано, тому що вона приймає SMILES і повертає одне числове значення.',
        '',
        '2. Вхідні та вихідні дані',
        'Вхід: SMILES молекули, наприклад CCO (етанол).',
        'Вихід: прогноз log10(S), де S — розчинність у моль/л.',
        'Що вище число, то вища прогнозована розчинність.',
        'Формулу перевіряє бібліотека RDKit.',
        '',
        '3. Результати аналізу',
        f'Кількість речовин із прогнозом: {len(ready)}.',
    ]
    if ready:
        for row in ready:
            report.append(f"{row['molecule_name']} ({row['formula']}): {float(row['predicted_value']):.4f} log10(моль/л)")
        maximum = max(ready, key=lambda row: float(row['predicted_value']))
        minimum = min(ready, key=lambda row: float(row['predicted_value']))
        report += [
            '',
            '4. Порівняння',
            'Найвища прогнозована розчинність: ' + maximum['molecule_name'],
            'Найнижча прогнозована розчинність: ' + minimum['molecule_name'],
        ]
    else:
        report.append('Запустіть main.py і виберіть пункт 2, щоб отримати реальні прогнози.')
    report += [
        '',
        '5. Висновок',
        'ШІ корисний для швидкої попередньої оцінки властивостей багатьох молекул.',
        'Проте результати моделі є прогнозами, а не експериментальними вимірюваннями.',
        'Точність залежить від набору навчальних даних, тому результат потребує перевірки.',
        '',
        '6. Запуск (Windows PowerShell)',
        'python -m pip install -r requirements.txt',
        'python main.py',
        'У меню вибрати 2, щоб проаналізувати 10 молекул і заповнити CSV.',
        'Під час першого запуску потрібен інтернет для завантаження моделі (~176 МБ).',
    ]
    REPORT_FILE.write_text('\n'.join(report) + '\n', encoding='utf-8')


def menu():
    rows = read_results()
    while True:
        print('\n========== AI CHEMISTRY LAB ==========')
        print('1. Добавить молекулу и сделать прогноз')
        print('2. Проанализировать 10 готовых молекул')
        print('3. Показать результаты')
        print('4. Отсортировать по растворимости')
        print('5. Найти молекулу по названию')
        print('6. Сравнить 2 или больше молекул')
        print('0. Выход')
        choice = input('Выберите пункт: ').strip()

        if choice == '0':
            break
        if choice == '1':
            name = input('Название: ').strip()
            formula = input('Формула (например, C2H6O): ').strip()
            smiles = input('SMILES (например, CCO): ').strip()
            try:
                if not name:
                    raise ValueError('Введите название.')
                add_result(rows, name, formula, smiles)
                write_report(rows)
            except Exception as error:
                print('Ошибка:', error)
        elif choice == '2':
            print('\nАнализ 10 молекул:')
            for name, formula, smiles in EXAMPLES:
                try:
                    add_result(rows, name, formula, smiles)
                except Exception as error:
                    print('Ошибка:', error)
                    break
            write_report(rows)
            print('CSV:', CSV_FILE.name)
        elif choice == '3':
            show_rows(ready_rows(rows))
        elif choice == '4':
            sorted_rows = sorted(ready_rows(rows), key=lambda row: float(row['predicted_value']), reverse=True)
            show_rows(sorted_rows)
        elif choice == '5':
            name = input('Введите название или часть названия: ').strip().casefold()
            found = [row for row in ready_rows(rows) if name in row['molecule_name'].casefold()]
            show_rows(found)
        elif choice == '6':
            names = input('Введите названия через запятую: ').split(',')
            found = []
            for name in names:
                for row in ready_rows(rows):
                    if row['molecule_name'].casefold() == name.strip().casefold():
                        if row not in found:
                            found.append(row)
            if len(found) < 2:
                print('Нужно указать не меньше двух уже проанализированных веществ.')
            else:
                show_rows(sorted(found, key=lambda row: float(row['predicted_value']), reverse=True))
        else:
            print('Нет такого пункта.')


if __name__ == '__main__':
    menu()
