import React, { Component } from 'react'
import activityBox from '../assets/buttons/activityBox.svg'

export class Box extends Component {
  render() {
    return (
      <div>
        <img src={activityBox} alt="activity" />
      </div>
    )
  }
}

export default Box