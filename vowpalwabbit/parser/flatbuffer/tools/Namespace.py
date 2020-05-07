# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Namespace(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNamespace(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Namespace()
        x.Init(buf, n + offset)
        return x

    # Namespace
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Namespace
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Namespace
    def Features(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Feature import Feature
            obj = Feature()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Namespace
    def FeaturesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Namespace
    def FeaturesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def NamespaceStart(builder): builder.StartObject(2)
def NamespaceAddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def NamespaceAddFeatures(builder, features): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(features), 0)
def NamespaceStartFeaturesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def NamespaceEnd(builder): return builder.EndObject()
