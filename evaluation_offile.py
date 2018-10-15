import click
import xlsxwriter
import numpy as np
import os

from model.utils.data_generator import DataGenerator
from model.utils.text_utils import diff_rows
from model.img2seq_ctc import Img2SeqCtcModel
from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.image import greyscale
from model.utils.general import minibatches

from evaluate_model import insert_image_worksheet, predict_with_image, read_img

model_name = ""
batch_size = 20

def eval_metric(expected_result, result):
    match_n, explanation = diff_rows(expected_result, result)
    match_n_1, explanation_1 = diff_rows(result, expected_result)

    return (match_n/len(expected_result) + match_n_1/len(result))/2, explanation

@click.command()
@click.option('--results', default="./results_from_server/results1/full/", help='Dir to results')
@click.option("--vocab", default="./results_from_server/results1/full/vocab.txt", help="vocab.txt")
@click.option("--prefix", default="./test_imgs/", help="prefix paths which used in souce file")
@click.option('--src_test', default="./test_imgs/src-test.txt", help = "Dir to source")
@click.option("--tgt_test", default="./test_imgs/tgt-test.txt", help = "Dir to target")
def main(results, vocab, prefix, src_test, tgt_test):
    assert os.path.isfile(src_test) and os.path.isfile(tgt_test)

    img_paths= [os.path.join(prefix, src.strip()) for src in open(src_test).readlines()]
    imgs     = [read_img(img_path)[-1] for img_path in img_paths]
    img_lbls = [_.strip() for _ in open(tgt_test).readlines()]

    predicts = predict_with_image(model_saved_path=results, vocab_path=vocab, imgs=imgs)

    workbook  = xlsxwriter.Workbook('result_from_%s.xlsx' % tgt_test.split('/')[-1].replace('.','_'))
    worksheet = workbook.add_worksheet("Accuracy Report")

    worksheet.write(0, 0, 'Index')
    worksheet.write(0, 1, 'File Name')
    worksheet.write(0, 2, 'Input')
    worksheet.write(0, 3, 'Output')
    worksheet.write(0, 4, 'CorrectAnswer')
    worksheet.write(0, 5, 'Accuracy')
    worksheet.write(0, 6, 'Explanation')

    scale = 5
    index = 0
    for _id, (lbl, predict) in enumerate(zip(img_lbls, predicts)):
        index += 1
        _row_idx = scale*index-scale+1
        acc, explanation = eval_metric(expected_result=lbl, result=predict)

        worksheet.write(_row_idx, 0, index)
        worksheet.write(_row_idx, 1, img_paths[_id])

        insert_image_worksheet(worksheet, img_paths[_id], index, 2, scale)
        worksheet.write(_row_idx, 3, predict)
        worksheet.write(_row_idx, 4, lbl)
        worksheet.write(_row_idx, 5, acc)
        worksheet.write(_row_idx, 6, explanation)

    workbook.close()
    #
    #
    #
    #
    #
    #
    # dir_output = results
    #
    # config_data = Config(dir_output + "data.json")
    # config_vocab = Config(dir_output + "vocab.json")
    # config_model = Config(dir_output + "model.json")
    #
    # vocab = Vocab(config_vocab)
    # model = Img2SeqCtcModel(config_model, dir_output, vocab)
    # model.build_pred()
    # model.restore_session(dir_output + "model.weights/")
    #
    # test_set = DataGenerator(path_formulas=config_data.path_formulas_test,
    #                          dir_images=config_data.dir_images_test, img_prepro=greyscale,
    #                          max_iter=config_data.max_iter, bucket=config_data.bucket_test,
    #                          path_matching=config_data.path_matching_test,
    #                          max_len=config_data.max_length_formula,
    #                          form_prepro=vocab.form_prepro, )
    #
    #
    # # initialize containers of references and predictions
    # if model._config.decoding == "greedy":
    #     refs, hyps = [], [[]]
    # elif model._config.decoding == "beam_search":
    #     refs, hyps = [], [[] for i in range(model._config.beam_size)]
    #
    # # iterate over the dataset
    # n_words, ce_words = 0, 0  # sum of ce for all words + nb of words
    # for img, formula in minibatches(test_set, batch_size):
    #     fd = model._get_feed_dict(img, training=False, formula=formula,
    #                              dropout=1)
    #     ce_words_eval, n_words_eval, ids_eval = model.sess.run(
    #         [model.ce_words, model.n_words, model.pred_test.ids],
    #         feed_dict=fd)
    #
    #     # TODO(guillaume): move this logic into tf graph
    #     if model._config.decoding == "greedy":
    #         ids_eval = np.expand_dims(ids_eval, axis=1)
    #
    #     elif model._config.decoding == "beam_search":
    #         ids_eval = np.transpose(ids_eval, [0, 2, 1])
    #
    #     n_words += n_words_eval
    #     ce_words += ce_words_eval
    #     for form, preds in zip(formula, ids_eval):
    #         refs.append(form)
    #         for i, pred in enumerate(preds):
    #             hyps[i].append(pred)
    #
    # # refs: ground-truth
    # # hyps: predicted
    #
    # # TODO: watch about hyps[0] ????, hyps[1] ???
    #



if __name__ == '__main__':
    main()