# coding=utf-8
import re
import jaconv
import unicodedata
import difflib


def correct_text(text):
    correct_dict = {
        '卜': ['ト'],
        '-': ['ー', '―', '‐', '-', '—', '一', '一'],
        '.': ['。', '．'],
        ',': ['、', '，'],
        '/': ['・', '／', '·', '・', '·'],
        '?': ['？', '❓'],
        '(': ['（', '〈', '【'],
        ')': ['）', '〉', '】'],
        '〇': ['o', 'O', '○', ],
    }
    remove_chars = ['〒', '※']

    for key, values in correct_dict.items():
        for value in values:
            text = text.replace(value, key)
    for c in remove_chars:
        text = text.replace(c, '')
    return text


def normalize_output_text(text):
    """Remove unusual characters from the ocr results

    :param text: ocr result need to be fix
    :returns: normalized string
    :rtype: string

    """
    output = []
    # text = re.sub("\f", "", text)
    if output.count(u"\n") > 3:
        text = re.sub(u"\t", "", text)
        text = re.sub(u"\r", "", text)
        text = re.sub(u"\v", "", text)
        text = re.sub(u" ", "", text)
    text = re.sub("\s+", "", text)
    text = text.strip()
    text = re.sub(u"（", u"(", text)
    text = re.sub(u"）", u")", text)
    text = re.sub(u"・", u".", text)
    text = re.sub(u"·", u".", text)
    text = re.sub(u"、", u",", text)
    text = re.sub(u"一", u"ー", text)
    text = re.sub(u"力", u"カ", text)
    text = re.sub(u".あり0なし", u".あり", text)
    text = re.sub(u"0あり.なし", u".なし", text)
    text = unicodedata.normalize('NFKC', text)
    text = jaconv.normalize(text, 'NFKC')
    '''
    Text correction
    '''
    text = re.sub(u"烹制限", u"無制限", text)
    for i, c in enumerate(text):
        if 9311 <= ord(c) and ord(c) < 9321:  # 0-9
            output.append(unichr(ord(c) + 48 - 9311))
        elif 9321 < ord(c) and ord(c) <= 9331:  # 10-19
            output.append("1")
            output.append(unichr(ord(c) + 48 - 9321))
        else:
            output.append(c)
    return ''.join(output)


def normalize_expected_text(key_text):
    """Remove unusual characters from the ocr results

    :param text: ocr result need to be fix
    :returns: normalized string
    :rtype: string

    """
    key_text = re.sub("（", "(", key_text)
    key_text = re.sub("）", ")", key_text)
    key_text = re.sub("／", "/", key_text)
    key_text = re.sub("・", ".", key_text)
    key_text = re.sub("※", "", key_text)
    key_text = re.sub("★", "", key_text)
    key_text = re.sub("\n", "", key_text)
    text = re.sub("\s+", "", key_text)
    text = re.sub("（", "(", text)
    text = re.sub("）", "(", text)
    text = re.sub("・", ".", text)
    text = re.sub("、", ",", text)
    text = re.sub(".あり0なし", ".あり", text)
    text = re.sub("0あり.なし", ".なし", text)
    '''
    Text correction
    '''
    text = re.sub("烹制限", "無制限", text)
    output = []
    for i, c in enumerate(text):
        if 9311 <= ord(c) and ord(c) < 9321:  # 0-9
            output.append(chr(ord(c) + 48 - 9311))
        elif 9321 < ord(c) and ord(c) <= 9331:  # 10-19
            output.append("1")
            output.append(chr(ord(c) + 48 - 9321))
        else:
            output.append(c)
    return ''.join(output)


# def diff_rows(s1, s2):
# diff = difflib.SequenceMatcher(None, s1, s2)
# match_n = 0
# for block in diff.get_matching_blocks():  match_n = match_n + block[2]
# return match_n


def normalize_text(text):
    text = re.sub('\s+', '', text)
    text = correct_text(text)
    text = unicodedata.normalize('NFKC', text)
    text = jaconv.normalize(text, 'NFKC')
    return text


def diff_rows(s1, s2):
    # s1 = self.preprocess_value(str(df['AI OCR']))
    # s2 = self.preprocess_value(str(df['Correct Answer']))
    diff = difflib.SequenceMatcher(None, s1, s2)
    result = ''
    format_pairs = ''
    # Hiển thị phần sai
    if s2 == '' and len(s1) > 0:
        result = 'OCR no value field'
    else:
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            print_s1 = 'SPACE' if s1[i1:i2] == ' ' else s1[i1:i2]
            print_s2 = 'SPACE' if s2[j1:j2] == ' ' else s2[j1:j2]
            print_s1 = 'ENTER' if s1[i1:i2] == '\n' else print_s1
            print_s2 = 'ENTER' if s2[j1:j2] == '\n' else print_s2
            if tag == 'replace':
                result = result + 'Mistake {} -> {}\n'.format(print_s2, print_s1)
                # format_pairs.append(('r', i1, i2))
                format_pairs = format_pairs + " ('r',{},{})".format(i1, i2)
            elif tag == 'delete':
                result = result + 'Excess {}(position {})\n'.format(print_s1, str(i1))
                # format_pairs.append(('r', i1, i2))
                format_pairs = format_pairs + " ('r',{},{})".format(i1, i2)
            elif tag == 'insert':
                if len(s1) == 0:
                    result = result + 'Can not OCR'
                else:
                    result = result + 'Lost {}(position {})\n'.format(print_s2, str(j1))
    ###
    match_n = 0
    for block in diff.get_matching_blocks():
        match_n = match_n + block[2]
    return match_n, result
