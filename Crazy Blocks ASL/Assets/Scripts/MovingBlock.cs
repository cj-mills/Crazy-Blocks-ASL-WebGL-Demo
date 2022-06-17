using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovingBlock : MonoBehaviour
{
    float moveSpeed = 2.5f;
    float heightRange = 1f;
    public static int score;
    public static int highScore = 0;

    float startingYPosition;

    Vector3 initPosition;

    public void ResetPosition()
    {
        score = 0;
        this.transform.position = initPosition;
    }


    // Start is called before the first frame update
    void Start()
    {
        initPosition = this.transform.position;
        startingYPosition = transform.position.y;
        score = 0;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        transform.position += Vector3.left * Time.deltaTime * moveSpeed;

        if (transform.position.x <= -15f)
        {
            transform.position += Vector3.right * 30f;
            float newY = startingYPosition + Random.Range(heightRange * -1, heightRange);
            transform.position = new Vector3(transform.position.x, newY, transform.position.z);
            score++;
            highScore = score > highScore ? score : highScore;
        }
    }
}
