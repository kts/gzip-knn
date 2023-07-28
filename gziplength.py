import zlib
import gzip

class GzipLengthCalc:
    """
    Efficient way to compute both,
      len(gzip.compress(data1))
    ane
      len(gzip.compress(data1 + b" " + data2))

    (and can be reused for multiple data2)

    Usage: (all data in bytes)

        g = GzipLengthCalc(data1)
        n1 = g.length1

        n2  = g.length2(data2)
        n2b = g.length2(data2b) # re-use for different data2

    """
    def __init__(self,data):
        """
        """
        #args copied from gzip.py:
        compress = zlib.compressobj(
            gzip._COMPRESS_LEVEL_BEST,
            zlib.DEFLATED,
            -zlib.MAX_WBITS,
            zlib.DEF_MEM_LEVEL,
            0)

        self.len_header_footer = 10 + 8
        
        #often '' (ie not flushed)
        self.clen1 = len(compress.compress(data))

        #store zlib state:
        self.compress1 = compress.copy()

        #finish:
        n = self.len_header_footer
        n += self.clen1
        n += len(compress.flush())

        # length1 = len(gzip.compress(data))
        self.length1 = n

        # we can do the " " here
        self.len2a = len(self.compress1.compress(b" "))

    def length2(self,data):
        """
        """
        c = self.compress1.copy()

        n = self.len_header_footer

        #lengths from first step:
        n += self.clen1 + self.len2a
        
        n += len(c.compress(data))
        n += len(c.flush())

        return n
