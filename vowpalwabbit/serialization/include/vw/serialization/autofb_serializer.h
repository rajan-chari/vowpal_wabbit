#pragma once

#include "vw/serialization/base.h"
#include "vw/serialization/autofb_schema.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/reflection.h"

namespace autofb
{
  using offset_of_any = flatbuffers::Offset<void>;

  class serializer
  {
  public:
    serializer(schema schema) : 
      _schema{schema}
    {
    }

    // inline offset_of_any write_flatbuffer(flatbuffers::FlatBufferBuilder& fbb, reflectable& target)
    // {
    //   reflector r{target};
    //   typesys::erased_lvalue_ref elv_target = r.reflect_scalar("this");

    //   return write_flatbuffer(fbb, elv_target);
    // }

    offset_of_any write_flatbuffer(flatbuffers::FlatBufferBuilder& fbb, typesys::erased_lvalue_ref& target);

    typesys::activation read_flatbuffer(const uint8_t* buf, typesys::erased_type target_type);

  private:
    schema _schema;
  };
}