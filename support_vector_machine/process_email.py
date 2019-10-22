import re


def process_email(email):
    cleantext = email.lower()
    # Remove HTML <> tags
    html = re.compile('<.*?>')
    cleantext = re.sub(html, ' ', cleantext)
    # Replace numbers
    nums = re.compile('\d+([., ]?\d*)*')
    cleantext = re.sub(nums, 'number ', cleantext)
    # Replace HTTP(s) links
    links = re.compile('^(http|https)://')
    cleantext = re.sub(links, 'httpaddr ', cleantext)
    # Replace email addresses
    email = re.compile('[^\s]+@[^\s]+')
    cleantext = re.sub(email, 'emailaddr ', cleantext)
    # Replace $ signs
    dollar = re.compile('[$]+')
    cleantext = re.sub(dollar, 'dollar ', cleantext)
    # Remove non-alphanumeric characters
    non_alpha_numeric = re.compile('[^0-9a-zA-Z]+')
    return re.sub(non_alpha_numeric, ' ', cleantext).strip()
