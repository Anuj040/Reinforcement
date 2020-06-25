from PIL import Image, ImageDraw, ImageFont
import numpy as np



def draw_sequences(i, k, step, action, draw, region_image, background, path_testing_folder, iou, reward,
                   gt_mask, region_mask, image_name, save_boolean):
    mask = Image.fromarray(255 * gt_mask)
    mask_img = Image.fromarray(255 * region_mask)
    image_offset = (1000 * step+100, 70)
    text_offset = (1000 * step+100, 550)
    masked_image_offset = (1000 * step+100, 1400)
    mask_offset = (1000 * step+100, 700)
    action_string = string_for_action(action)
    footnote = 'action: ' + action_string + ' ' + 'reward: ' + str(reward) + ' Iou:' + str(iou)
    draw.text(text_offset, str(footnote), (0, 0, 0))
    img_for_paste = Image.fromarray(region_image)
    background.paste(img_for_paste, image_offset)
    background.paste(mask, mask_offset)
    background.paste(mask_img, masked_image_offset)
    file_name = path_testing_folder + '/' + image_name + '_' + str(i) + '_object_' + str(k) + '.png'
    if save_boolean == 1:
        background.save(file_name)
    return background

def draw_sequences_test(step, action, qval, draw, region_image, background, path_testing_folder,
                        region_mask, image_name, save_boolean):
    aux = np.asarray(region_image, np.uint8)
    img_offset = (1000 * step+100, 70)
    footnote_offset = (1000 * step+100, 550)
    q_predictions_offset = (1000 * step+100, 500)
    mask_img_offset = (1000 * step+100, 700)
    img_for_paste = Image.fromarray(aux)
    background.paste(img_for_paste, img_offset)
    mask_img = Image.fromarray(255 * region_mask)
    background.paste(mask_img, mask_img_offset)
    footnote = 'action: ' + str(action)
    q_val_predictions_text = str(qval)
    draw.text(footnote_offset, footnote, (0, 0, 0))
    draw.text(q_predictions_offset, q_val_predictions_text, (0, 0, 0))
    file_name = path_testing_folder + '/'  + image_name + '.png'

    if save_boolean:
        background.save(file_name)
    return background
                
def string_for_action(action):
    action_list = {0: "START",
              1: 'up-left',
              2: 'up-right', 
              3: 'down-left',
              4: 'down-right',
              5: 'center',
              6: 'TRIGGER'}
    return action_list[action]