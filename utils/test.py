from cat import cat
from collections import namedtuple


def main():
    # initialize a new cat
    ct = cat()
    # set the attribute
    Paar = namedtuple('Paar', ['c1','c2'])
    paars = []
#    paars.append(Paar(c1 = '/tmp/de.en2014decoder_hidden.ref', c2 = '/tmp/de.en2015decoder_hidden.ref'))
    paars.append(Paar(c1 = '/tmp/de.en2014decoder_hidden.s1', c2 = '/tmp/de.en2015decoder_hidden.s1'))
    paars.append(Paar(c1 = '/tmp/de.en2014decoder_hidden.s2', c2 = '/tmp/de.en2015decoder_hidden.s2'))
    # concat the paar
    for paar in paars:
        ct.forward(paar.c1, paar.c2)

if __name__ == '__main__':
    main()
