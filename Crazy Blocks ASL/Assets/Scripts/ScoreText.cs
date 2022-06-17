using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScoreText : MonoBehaviour
{
    // Update is called once per frame
    void Update()
    {
        GetComponent<TMP_Text>().SetText($"Score: {MovingBlock.score}");
    }
}
