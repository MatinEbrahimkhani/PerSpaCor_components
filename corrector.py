from .corpus import Type
from .labeler.labeler import Labeler

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class PerSpaCor:
    def __init__(self, model_name):
        self.models = {'bert-multilingual': "PerSpaCor/bert-base-multilingual-uncased",
                       'parsbert': "PerSpaCor/HooshvareLab-bert-base-parsbert-uncased",
                       'charbert': "PerSpaCor/imvladikon-charbert-roberta-wiki",
                       'parsroberta': "PerSpaCor/HooshvareLab-roberta-fa-zwnj-base"}

        model= self.models[model_name]
        self.__model_path = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForTokenClassification.from_pretrained(model, num_labels=3)

        self._labeler = Labeler(tags=(1, 2),
                                regexes=(r'[^\S\r\n\v\f]', r'\u200c'),
                                chars=(" ", "‌"),
                                class_count=2)

    def _tokenize(self, chars, chunk_size=512):
        input_ids = []

        ids_ = [101] + [self._tokenizer.encode(char)[1] for char in chars] + [102]
        for i in range(0, len(ids_), chunk_size):
            chunked_ids = ids_[i:i + chunk_size]
            input_ids.append(chunked_ids)
        return input_ids

    def correct(self, text, report=True):
        chars, labels = self._labeler.label_text(text, corpus_type=Type.whole_raw)
        input_ids = self._tokenize(chars, chunk_size=512)
        input_ids = torch.tensor(input_ids)

        labels = [0] + labels + [0]
        logits = self._model(input_ids).logits
        predicted_labels = torch.argmax(logits, dim=-1)
        return self._labeler.text_generator([' '] + chars + [' '],
                                            predicted_labels,
                                            corpus_type=Type.whole_raw)


# corrector = PerSpaCor('bert-multilingual')
#
# correction = corrector.correct(
#     "همه روزه، از ابزارهایی استفاده می‎کنیم و بدون توجه به شگفتی و تاثیر آن‌ها در زندگی روزمره به راحتی آن‌ها را  نادیده می‌گیریم. یکی از این ابزارهای روزمره و البته پر قدرت، زبان است که هر روز به وسیله آن با اطرافیان ارتباط برقرار می‌کنیم، فکر می‌کنیم و احساساتمان را بیان می‌کنیم. زبان طبیعی از ماقبل تاریخ همراه ما بوده و هم پای ما دست خوش تغییرات بسیاری شده‌است. این ابزار، اکنون به قدری کارآمد و منعطف است که به حافظ امکان سرودن شعر می‌دهد، به حقوق‌دان‌ها امکان نوشتن قوانین اجتماعی را می‌دهد و به طنازان امکان لطیفه‌گویی می‌بخشد.")
# # correction = corrector.correct(
# # "خوانایی هر متنی مهم دلیل تحریر آن است با این وجود ممکن است اشتباهاتی از سوی نویسنده باعث کاهش خوانایی متن می شود. این اشتباهات را در دو دسته ی معنایی و نحوی تقسیم می‌شود. اشتباه‌های معنایی شامل بیان جمله به شکل نادرست، انتخاب کلمه های غلط که باعث کاهش درک مطلب و کج فهمی مخاطب می شوند. و اشتباه های نحوی که شامل غلط املایی، عدم استفاده و یا استفاده ناصحیح از علائم اشاره مانند نقطه، ویرگول، پرانتز و غیره، پاراگراف بندی ناصحیح و همچنین فاصله گذاری کلمات و بین کلمات و یا فاصله های مربوط به علائم اشاره می‌شوند. ",report=True)
# print(correction)
# # corrector.test()
