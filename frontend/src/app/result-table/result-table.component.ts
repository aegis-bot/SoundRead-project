import { Component, Input, OnInit } from '@angular/core';


@Component({
  selector: 'app-result-table',
  templateUrl: './result-table.component.html',
  styleUrls: ['./result-table.component.css']
})

export class ResultTableComponent implements OnInit {
  @Input() predictedMelody: string;
  @Input() predictedLyrics: string = "No lyrics found.";
  
  constructor() { }

  ngOnInit(): void {
  }

  //public serverAdd = "http://127.0.0.1:5000/static/temp.mid";
  
  playMidiCall() {
    let serverAdd = this.predictedMelody;
    let midiTemplate = "MIDIjs.play('" + serverAdd + "');";

    var js = midiTemplate;
    eval(js);
  }

  playAudio() {
  
    if(!!this.predictedMelody) {
      console.log("play!")
      
    } else {
      console.log("file not in results yet")
    }
    //this.audioObject.play
  }
}
