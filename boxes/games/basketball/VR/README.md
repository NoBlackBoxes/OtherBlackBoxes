# Games : basketball : VR

Notes for setting up Occulus Quest (1/2) for use in Unity with Linux

- Make sure you have a late(ish) version of Unity
- JDK and NDK errors (make sure progams used have correct permissions: e.g. chmod +x llvm-strip)
- Need Occulus Integration from Asset store
- Enable XR plugin

## First project
- Create a scene
- Add OVRCameraRig as the main camera
- Make sure your new scene is included in the build
- (add textures to model)
- Alter lighting environment (windows-rendering-environment) as needed


## Getting inputs

Add the following script to a spot light game object, attach the light to one of the controllers in the OVRCameraRig

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OVR_test_trigger : MonoBehaviour
{
    Light light;
    float amplitude = 100.0f;

    void Update()
    {
        light.range = amplitude * OVRInput.Get(OVRInput.RawAxis1D.RIndexTrigger);;
    }
}
```

Add the following to a sphere object (and connect to the left hand controller object)

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OVR_ball_trigger : MonoBehaviour
{
    public GameObject hand;

    void Update()
    {
        if (OVRInput.Get(OVRInput.RawAxis1D.LIndexTrigger) > 0.5f)
        {
            transform.position = hand.transform.position;
            GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
            GetComponent<Rigidbody>().angularVelocity = new Vector3(0f, 0f, 0f);
        }
    }
}
```

Add mesh collidaer to scene