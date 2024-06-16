from fastchat.model import get_conversation_template


def load_conversation_template(template_name):
    conv_template = get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


def generate_input(conv_template,prompt,adv_suffix=None):   
    conv_template.messages = []
    if adv_suffix is not None:
        conv_template.append_message(conv_template.roles[0], f"{prompt} {adv_suffix}")
        conv_template.append_message(conv_template.roles[1], None)
        result = conv_template.get_prompt()
    else:
        conv_template.append_message(conv_template.roles[0], f"{prompt}.")
        conv_template.append_message(conv_template.roles[1], None)
        result = conv_template.get_prompt() + " "
    return result