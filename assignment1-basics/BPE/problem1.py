#1.返回NULL控制字符
print(len(chr(0)))

#2.repr显示转义序列'\x00'，而print会对之进行渲染，使之不可见
print(repr(chr(0)))

#3.

print("this is a test" + chr(0) + "string")  #打印出来：在字符串中一个合法而不可见的字符
