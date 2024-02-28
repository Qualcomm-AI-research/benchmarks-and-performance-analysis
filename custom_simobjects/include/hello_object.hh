# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

#ifndef __CUSTOM_SIMOBJECTS_HELLO_OBJECT_HH__
#define __CUSTOM_SIMOBJECTS_HELLO_OBJECT_HH__

#include "params/HelloObject.hh" 
#include "sim/sim_object.hh" 

namespace gem5 
{

    class HelloObject : public SimObject 
    {
        public: 
            HelloObject(const HelloObjectParams &p); 
    }; 
} // namespace gem5 

#endif  // __CUSTOM_SIMOBJECT_HELLO_OBJECT_HH__
