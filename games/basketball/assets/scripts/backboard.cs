using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class backboard : MonoBehaviour
{
    public AudioSource backboard_source;
 
    // Start is called before the first frame update
    void Start()
    {
        backboard_source = GetComponent<AudioSource>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    
    void OnCollisionEnter (Collision collision)
    {
        // Get collision speed and set sound volume
        backboard_source.volume = Mathf.Clamp01(collision.relativeVelocity.magnitude / 20);

        // Play sound effect
        backboard_source.Play();
    }

}
