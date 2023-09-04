# Games : basketball

Create a basketball game in Unity

## Steps

- (Update Unity)
- Create a new Unity project called "basketball"
- Import Blender model of the basketball (Assets-Import new)
  - Unpack "prefab" and remove everything but the ball
- Add "rigid body" (Physics - for gravity)
- Add "sphere" collider to ball (Physics - for collisions)
- Add new physics "material" for collider (for friction, bounciness, etc.)

## Dribbling
- Create a new script called "dribble" with these contents:

```c#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class dribble : MonoBehaviour
{    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown("d"))
        {
            float thrust = -1000.0f;
            
            GetComponent<Rigidbody>().AddRelativeForce(0f,thrust,0f,ForceMode.Acceleration);
        }
    }
}
```

- Now add audio source (for bounce sound) to ball object (untick "play on awake")
- Add this audio source to your dribble script

```c#
  public AudioSource bounce_source;

  // In start functions
  bounce_source = GetComponent<AudioSource>()

  // Add this function (for collision handling)
  void OnCollisionEnter (Collision collision)
  {
    // Play sound effect
    bounce_source.Play();
  }

```

- Make the sound dependent on collision speed

```c#
  // Get collision speed and set sound volume
  bounce_source.volume = Mathf.Clamp01(collision.relativeVelocity.magnitude / 20);
```

## Resetting

- If things get messy, then press "r" to retrieve ball

```c#
if(Input.GetKeyDown("r"))
{    
    transform.position = new Vector3(0f, 10f, 0f);
    GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
}
```

## Shooting

- Add a "Shot Camera" (GameObject)
- Press "h" to hold the ball
- When shooting, take a first person persoective around the ball to choose trajectory...
 - Add another camera ("shot camera")
 - Make it a child of the "Ball"
 - Add a toggle button in script (that also disables physics)
 - Add "mouse look" rotation when in shooting mode

 ```c#
  Vector2 rotation = Vector2.zero;
  
  // In update

  // If shooting, mouse look aiming
  if(shooting)
  {
      rotation.y += Input.GetAxis ("Mouse X");
      rotation.x += -Input.GetAxis ("Mouse Y");
      transform.eulerAngles = (Vector2)rotation;
  }

- Press "s" again to launch shot

```c#
  Vector3 shot_vector = new Vector3(0.0f, 0.0f, 1000.0f);                            
  GetComponent<Rigidbody>().AddRelativeForce(shot_vector);
  GetComponent<Rigidbody>().useGravity = true;
```

 ```