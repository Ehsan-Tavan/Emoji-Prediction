#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-msg=too-many-boolean-expressions
# pylint: disable-msg=anomalous-backslash-in-string

"""
normalizer.py is writen for normalizing tweets
"""
import re
import hazm

__author__ = "Ehsan Tavan"
__project__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "10/06/2020"


class Normalizer:
    """
    Normalizer class is written for normalizing text
    """
    def __init__(self):
        self.hazm_normalizer = hazm.Normalizer()

    @staticmethod
    def remove_useless_chars(input_text):
        """
        remove bad characters
        :param input_text: input sentence
        :return: clean sentence
        """
        output_text = ""
        for i in input_text:
            # arabic-persian
            # spaces
            # next lines
            # numbers
            # symbols
            if (1536 <= ord(i) <= 1791) or (1872 <= ord(i) <= 1919) or \
                    (2208 <= ord(i) <= 2303) or (64336 <= ord(i) <= 65023) or \
                    (65136 <= ord(i) <= 65279) or (126464 <= ord(i) <= 126719) or \
                    (ord(i) == 32) or (ord(i) == 160) or (ord(i) == 5760) or \
                    (ord(i) == 6158) or (ord(i) == 8192) or (ord(i) == 8193) or \
                    (ord(i) == 8194) or (ord(i) == 8195) or (ord(i) == 8196) or \
                    (ord(i) == 8197) or (ord(i) == 8198) or (ord(i) == 8199) or \
                    (ord(i) == 8200) or (ord(i) == 8201) or (ord(i) == 8202) or \
                    (ord(i) == 8203) or (ord(i) == 8204) or (ord(i) == 8239) or \
                    (ord(i) == 8287) or (ord(i) == 12288) or (ord(i) == 65279) or \
                    (ord(i) == 133) or (ord(i) == 10) or (ord(i) == 11) or \
                    (ord(i) == 12) or (ord(i) == 13) or (ord(i) == 8232) or \
                    (ord(i) == 8233) or (48 <= ord(i) <= 57) or \
                    (1632 <= ord(i) <= 1641) or (1776 <= ord(i) <= 1785) or \
                    (65296 <= ord(i) <= 65305) or (120782 <= ord(i) <= 120831) or \
                    (32 <= ord(i) <= 47) or (58 <= ord(i) <= 64) or \
                    (91 <= ord(i) <= 95) or (123 <= ord(i) <= 126):
                output_text += i
        return output_text

    @staticmethod
    def arabic_to_farsi(input_text):
        """
        change arabic character to persian
        :param input_text: input sentence
        :return: clean sentence
        """
        # [........... , standard uniode]
        unicode_list = [
            [65153, 1570],

            [65166, 65165, 65160, 65159, 65156, 65155, 65154, 64829, 64828, 64337,
             64336, 1908, 1907, 1653, 1651, 1650, 1649, 1573, 1571, 1575],

            [65170, 65169, 65168, 65167, 64349, 64348, 64347, 64341, 64340, 64339,
             64338, 2230, 2209, 2208, 1878, 1877, 1876, 1875, 1874, 1873, 1872, 1664,
             1659, 1646, 1576],

            [64345, 64344, 64343, 64342, 2231, 1662],

            [65180, 65179, 65178, 65177, 65176, 65175, 65174, 65173, 65172, 65171,
             64361, 64360, 64359, 64358, 64357, 64356, 64355, 64354, 64353, 64352,
             64351, 64350, 2232, 1663, 1661, 1660, 1658, 1657, 1578],

            [1579],

            [65184, 65183, 65182, 65181, 65180, 64377, 64376, 64375, 64374, 64373,
             64372, 64371, 64370, 2210, 1580],

            [64385, 64384, 64383, 64382, 64381, 64380, 64379, 64378, 1727, 1671, 1670],

            [65188, 65187, 65186, 65185, 1916, 1906, 1903, 1902, 1880, 1879, 1669, 1668,
             1667, 1666, 1665, 1581],

            [65192, 65191, 65190, 65189, 1582],

            [65194, 65193, 64393, 64392, 64391, 64390, 64389, 64388, 64387, 64386, 2222,
             1882, 1881, 1774, 1680, 1679, 1678, 1677, 1676, 1675, 1674, 1673, 1672, 1583],

            [65196, 65195, 64604, 64603, 1584],

            [65198, 65197, 64397, 64396, 2233, 2218, 1905, 1900, 1899, 1883, 1775, 1689,
             1687, 1686, 1685, 1684, 1683, 1682, 1681, 1585],

            [65200, 65199, 2226, 1586],

            [64395, 64394, 1688],

            [65204, 65203, 65202, 65201, 65200, 1918, 1917, 1904, 1901, 1884,
             1692, 1691, 1690, 1587],

            [65208, 65207, 65206, 65205, 1786, 1588],

            [65212, 65211, 65210, 65209, 2223, 1694, 1693, 1589],

            [65216, 65215, 65214, 65213, 1787, 1590],

            [65220, 65219, 65218, 65217, 2211, 1695, 1591],

            [65224, 65223, 65222, 65221, 1592],

            [65228, 65227, 65226, 65225, 2227, 1887, 1886, 1885, 1788, 1696, 1593],

            [65232, 65231, 65230, 65229, 1594],

            [65236, 65235, 65234, 65233, 64369, 64368, 64367, 64366, 64365, 64364, 64363,
             64362, 2235, 2212, 1888, 1702, 1701, 1700, 1699, 1698, 1697, 1601],

            [65240, 65239, 65238, 65237, 2236, 2213, 1704, 1703, 1647, 1602],

            [65244, 65243, 65242, 65241, 64470, 64469, 64468, 64467, 64401, 64400, 64399,
             64398, 2228, 1919, 1892, 1891, 1890, 1710, 1709, 1708, 1707, 1706,
             1603, 1596, 1595, 1705],

            [64413, 64412, 64411, 64410, 64409, 64408, 64407, 64406, 64405, 64404, 64403,
             64402, 2224, 1716, 1715, 1714, 1713, 1712, 1711],

            [65248, 65247, 65246, 65245, 2214, 1720, 1719, 1718, 1717, 1604],

            [65252, 65251, 65250, 65249, 2215, 1894, 1893, 1605],

            [65256, 65255, 65254, 65253, 64419, 64418, 64417, 64416, 64415, 64414, 2237,
             1897, 1896, 1895, 1725, 1724, 1723, 1722, 1721, 1606],

            [65262, 65261, 65158, 65157, 64483, 64482, 64481, 64480, 64479, 64478, 64477,
             64476, 64475, 64474, 64473, 64472, 64471, 2219, 1913, 1912, 1743, 1739, 1738,
             1737, 1736, 1735, 1734, 1733, 1732, 1655, 1654, 1572, 1608],

            [65260, 65259, 65258, 65257, 64429, 64428, 64427, 64426, 64425, 64424, 64423,
             64422, 64421, 64420, 1791, 1749, 1731, 1730, 1729, 1728, 1726, 1577, 1607],

            [65268, 65267, 65266, 65265, 65264, 65263, 65164, 65163, 65162, 65161, 64617,
             64616, 64605, 64516, 64515, 64511, 64510, 64509, 64508, 64507, 64506, 64505,
             64504, 64503, 64502, 64502, 64502, 64502, 64502, 64489, 64488, 64487, 64486,
             64485, 64484, 64433, 64432, 64431, 64430, 2234, 2217, 2216, 1915, 1914, 1911, 1910,
             1909, 1747, 1746, 1745, 1744, 1742, 1741, 1656, 1610, 1609, 1599, 1598, 1597, 1574,
             1568, 1574, 1740],

            [1563],

            [1567],

            [1642]
        ]
        output_text = ""
        for k in input_text:
            for alph in unicode_list:
                if ord(k) in alph:
                    output_text += chr(alph[-1])
            if (ord(k) == 32) or (ord(k) == 160) or (ord(k) == 5760) or (ord(k) == 6158) or \
                    (ord(k) == 8192) or (ord(k) == 8193) or \
                    (ord(k) == 8194) or (ord(k) == 8195) or \
                    (ord(k) == 8196) or (ord(k) == 8197) or \
                    (ord(k) == 8198) or (ord(k) == 8199) or \
                    (ord(k) == 8200) or (ord(k) == 8201) or \
                    (ord(k) == 8202) or (ord(k) == 8203) or \
                    (ord(k) == 8204) or (ord(k) == 8239) or \
                    (ord(k) == 8287) or (ord(k) == 12288) or \
                    (ord(k) == 65279) or (ord(k) == 133) or \
                    (ord(k) == 10) or (ord(k) == 11) or \
                    (ord(k) == 12) or (ord(k) == 13) or \
                    (ord(k) == 8232) or (ord(k) == 8233) or \
                    (48 <= ord(k) <= 57) or (1632 <= ord(k) <= 1641) or \
                    (1776 <= ord(k) <= 1785) or \
                    (65296 <= ord(k) <= 65305) or (120782 <= ord(k) <= 120831) or \
                    (32 <= ord(k) <= 47) or \
                    (58 <= ord(k) <= 64) or (91 <= ord(k) <= 95) or \
                    (123 <= ord(k) <= 126):
                output_text += k
        return output_text

    @staticmethod
    def remove_some_chars(input_text):
        """
        remove some characters
        :param input_text: input sentence
        :return: clean sentence
        """
        my_tweet = input_text.replace(chr(10), chr(32))  # \n
        my_tweet = my_tweet.replace(chr(1567), chr(32))  # ؟
        my_tweet = my_tweet.replace(chr(1563), chr(32))  # ؛
        my_tweet = my_tweet.replace(chr(1642), chr(32))  # ٪
        my_tweet = my_tweet.replace(chr(160), chr(32))
        my_tweet = my_tweet.replace(chr(8202), chr(32))  # Hair Space 8203
        my_tweet = my_tweet.replace(chr(8203), chr(32))  # Zero Width Space
        my_tweet = my_tweet.replace(chr(42), chr(32))  # *
        my_tweet = my_tweet.replace(chr(126), chr(32))  # ~
        my_tweet = my_tweet.replace(chr(95), chr(32))  # _
        my_tweet = my_tweet.replace(chr(47), chr(32))  # /
        my_tweet = my_tweet.replace(chr(94), chr(32))  # ^
        my_tweet = my_tweet.replace(chr(95), chr(32))  # _
        my_tweet = my_tweet.replace(chr(60), chr(32))  # >
        my_tweet = my_tweet.replace(chr(62), chr(32))  # >
        my_tweet = my_tweet.replace(chr(40), chr(32))  # )
        my_tweet = my_tweet.replace(chr(41), chr(32))  # (
        my_tweet = my_tweet.replace(chr(43), chr(32))  # +
        my_tweet = my_tweet.replace(chr(1642), chr(32))  # ٪
        my_tweet = my_tweet.replace(chr(1642), chr(32))  # »
        my_tweet = my_tweet.replace(chr(171), chr(32))  # «
        my_tweet = my_tweet.replace(chr(125), chr(32))  # }
        my_tweet = my_tweet.replace(chr(123), chr(32))  # {
        my_tweet = my_tweet.replace(chr(124), chr(32))  # |
        my_tweet = my_tweet.replace(chr(91), chr(32))  # [
        my_tweet = my_tweet.replace(chr(93), chr(32))  # ]
        my_tweet = my_tweet.replace(chr(92), chr(32))  # \\
        my_tweet = my_tweet.replace(chr(61), chr(32))  # =
        my_tweet = my_tweet.replace(chr(45), chr(32))  # -
        my_tweet = my_tweet.replace(chr(39), chr(32))  # '
        my_tweet = my_tweet.replace(chr(34), chr(32))  # "
        return my_tweet

    def normalizer_text(self, input_text):
        """
        main function for normalizer class
        :param input_text: input sentence
        :return: clean sentence
        """
        input_text = input_text.rstrip('\r\n').strip()
        normalized_text = input_text.lower()
        # Extract @somebody:
        normalized_text = re.sub('@[^\s]+', '', normalized_text)
        normalized_text = re.sub("@([^@]{0,30})\s", "", normalized_text)
        normalized_text = re.sub("@([^@]{0,30})）", "", normalized_text)
        # Remove links
        normalized_text = re.sub(r"http\S+", "", normalized_text)
        normalized_text = re.sub("http://[a-zA-z./\d]*", "", normalized_text)
        # remove english characters
        normalized_text = re.sub("[a-z]+", "", normalized_text)
        # Remove punctuation
        normalized_text = re.sub(r"[،,:.;@#?!&$]+\ *", " ", normalized_text)
        # Remove digits
        normalized_text = re.sub("\d", "", normalized_text)
        # Remove ....
        normalized_text = re.sub("\.*", "", normalized_text)
        # Remove multiple spaces
        normalized_text = re.sub("\s\s+", " ", normalized_text)
        # Remove more than 2 times repeat of character
        normalized_text = re.sub(r'(.)\1+', r'\1\1', normalized_text)
        # Remove useless characters
        normalized_text = self.remove_useless_chars(normalized_text)
        # Change arabic character to farsi character
        normalized_text = self.arabic_to_farsi(normalized_text)
        # Remove some characters
        normalized_text = self.remove_some_chars(normalized_text)
        # Remove multiple spaces
        normalized_text = re.sub("\s\s+", " ", normalized_text)
        # Normalize with hazm
        normalized_text = self.hazm_normalizer.normalize(normalized_text)
        return normalized_text.strip()
