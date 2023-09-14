def get_roi_prostate(img_input):
    return [
        [img_input[697:700, 638:655], img_input[718:721, 638:655]],
        [img_input[455:462, 874:882], img_input[469:476, 893:900]],
        [img_input[355:368, 616:629], img_input[404:417, 605:618]]
    ]


def get_roi_carotis(img_input):
    return [
        [img_input[529:533, 79:90], img_input[543:547, 79:90]],
        [img_input[413:418, 241:253], img_input[427:432, 241:253]],
        [img_input[259:270, 297:322], img_input[288:299, 295:320]]
    ]
