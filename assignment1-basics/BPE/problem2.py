s = "Miles and 黎子姗"
#将每个字符编码成若干字节
#每个字节取值范围：0-255

#基础词表只需要256个符号，不必担心OOV
#但是可能会出现长序列
utf8_encoded = s.encode("utf-8")
utf16_encoded = s.encode("utf-16")
utf32_encoded = s.encode("utf-32")
print(utf8_encoded)
#每个字节
print(list(utf8_encoded))
print(list(utf16_encoded))
print(list(utf32_encoded))
#utf16和utf32的序列长度会很长(前者大部分时候2字节，后者固定4字节，有很多0，出现冗余)
#utf8更加紧凑，序列更短，现实世界常用，利于训练。


#2.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])#逐字节解码，但是可能一个字符对应多个字节，就会出错，
#print(decode_utf8_bytes_to_str_wrong(utf8_encoded)) #bytes

#'utf-8' codec can't decode byte 0xe9 in position 0: unexpected end of data
#一个字符可能跨越多个字节，孤立逐字节解码是可能导致错误

#3.非法utf-8字节:utf-8' codec can't decode byte 0xff in position 0: invalid start byte
b'\xff\xff'.decode()
#有一些字节是无效的utf-编码